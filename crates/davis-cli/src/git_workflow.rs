use std::path::Path;
use std::process::Command;

pub(crate) fn verify_operator_worktree(
    repository: &Path,
    dataset_ids: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let branch = git_output(repository, &["branch", "--show-current"])?;
    validate_personal_branch(&branch)?;
    git_run(repository, &["fetch", "origin", "main"])?;
    let current = Command::new("git")
        .args(["merge-base", "--is-ancestor", "origin/main", "HEAD"])
        .current_dir(repository)
        .status()?;
    if !current.success() {
        return Err(
            "personal branch does not contain the latest origin/main; update it with `git merge --ff-only origin/main` before editing"
                .into(),
        );
    }
    let tracked = git_output(
        repository,
        &["diff", "--name-only", "-z", "--no-renames", "HEAD", "--"],
    )?;
    let untracked = git_output(
        repository,
        &["ls-files", "--others", "--exclude-standard", "-z", "--"],
    )?;
    for path in tracked
        .split('\0')
        .chain(untracked.split('\0'))
        .filter(|path| !path.is_empty())
    {
        if !path_belongs_to_datasets(path, dataset_ids) {
            return Err(format!(
                "uncommitted change outside the selected dataset must be resolved before push: {path}"
            )
            .into());
        }
    }
    Ok(())
}

pub(crate) fn commit_and_push_operator_changes(
    repository: &Path,
    dataset_ids: &[String],
    selected_dataset: Option<&str>,
    message: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut add = Command::new("git");
    add.arg("add").arg("-A").arg("--");
    for dataset_id in dataset_ids {
        add.arg(format!(".davis/datasets/{dataset_id}.yaml"));
        add.arg(format!("data/{dataset_id}"));
    }
    if !add.current_dir(repository).status()?.success() {
        return Err("git add failed".into());
    }
    let staged = !Command::new("git")
        .args(["diff", "--cached", "--quiet"])
        .current_dir(repository)
        .status()?
        .success();
    if !staged {
        println!("Git changes: none");
        return Ok(());
    }
    let default_message = selected_dataset.map_or_else(
        || "data: update all datasets".to_owned(),
        |dataset_id| format!("data: update {dataset_id}"),
    );
    let message = message.unwrap_or(&default_message);
    if !Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(repository)
        .status()?
        .success()
    {
        return Err("git commit failed".into());
    }
    let branch = git_output(repository, &["branch", "--show-current"])?;
    if !Command::new("git")
        .args(["push", "--set-upstream", "origin", &branch])
        .current_dir(repository)
        .status()?
        .success()
    {
        return Err(
            "git push failed; the local commit is preserved and can be pushed again".into(),
        );
    }
    println!("Git branch pushed: {branch}");
    println!("Open a Pull Request to main; catalog publication remains unchanged");
    Ok(())
}

pub(crate) fn verify_publish_git_state(
    repository: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let branch = git_output(repository, &["branch", "--show-current"])?;
    if branch != "main" {
        return Err(format!(
            "catalog publication is allowed only from main; current branch is {branch}"
        )
        .into());
    }
    if !git_output(repository, &["status", "--porcelain"])?.is_empty() {
        return Err("catalog publication requires a clean Git working tree".into());
    }
    git_run(repository, &["fetch", "origin", "main"])?;
    let head = git_output(repository, &["rev-parse", "HEAD"])?;
    let origin_main = git_output(repository, &["rev-parse", "origin/main"])?;
    if head != origin_main {
        return Err(
            "local main does not match origin/main; run `git pull --ff-only` and retry".into(),
        );
    }
    Ok(())
}

pub(crate) fn git_output(
    repository: &Path,
    arguments: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .output()?;
    if !output.status.success() {
        return Err(format!("git {} failed", arguments.join(" ")).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn git_run(repository: &Path, arguments: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .status()?;
    if !status.success() {
        return Err(format!("git {} failed", arguments.join(" ")).into());
    }
    Ok(())
}

fn validate_personal_branch(branch: &str) -> Result<(), String> {
    if branch.is_empty() {
        return Err("davis push requires a named personal working branch; HEAD is detached".into());
    }
    if branch == "main" {
        return Err(
            "davis push must not run from main; switch to a personal working branch first".into(),
        );
    }
    Ok(())
}

fn path_belongs_to_datasets(path: &str, dataset_ids: &[String]) -> bool {
    dataset_ids.iter().any(|dataset_id| {
        path == format!(".davis/datasets/{dataset_id}.yaml")
            || path.starts_with(&format!("data/{dataset_id}/"))
    })
}

#[cfg(test)]
mod tests {
    use super::{path_belongs_to_datasets, validate_personal_branch};

    #[test]
    fn push_accepts_any_named_branch_except_main() {
        for valid in ["operator/alice-1", "data-update", "team/network/2026"] {
            assert!(validate_personal_branch(valid).is_ok(), "{valid}");
        }
        assert!(validate_personal_branch("main").is_err());
        assert!(validate_personal_branch("").is_err());
    }

    #[test]
    fn only_selected_dataset_paths_are_allowed_in_operator_commit() {
        let datasets = vec!["network/tokyo-metropolitan-area".to_owned()];
        assert!(path_belongs_to_datasets(
            "data/network/tokyo-metropolitan-area/link.csv.schema.yaml",
            &datasets,
        ));
        assert!(path_belongs_to_datasets(
            ".davis/datasets/network/tokyo-metropolitan-area.yaml",
            &datasets,
        ));
        assert!(!path_belongs_to_datasets("README.md", &datasets));
        assert!(!path_belongs_to_datasets(
            "data/network/matsuyama/link.csv",
            &datasets,
        ));
    }
}
