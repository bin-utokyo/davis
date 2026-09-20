# Davis Site Admin Operations Guide

[日本語](site-admin-installation.md)

This guide is for Site Admins who manage access groups and dataset download permissions for an entire Davis Web deployment. For routine data and schema updates, use the [Installation Guide for Organizers](operator-installation_en.md). For data retrieval only, use the [Installation Guide for Participants](participant-installation_en.md).

## 1. Role boundaries

| Role | Credential | Main operations |
| --- | --- | --- |
| Participant | Participant code for one access group | Search the public catalog and download permitted datasets |
| Organizer | Organizer code for the same access group | Upload objects, claim new datasets, and publish the catalog |
| Site Admin | Deployment-wide Site Admin code | Create access groups, change dataset grants, and migrate existing storage |
| Cloudflare administrator | Cloudflare Account authentication | Configure and deploy the Worker, Secrets, and R2 bindings |

A Site Admin has stronger permission-management and storage-migration privileges than a routine organizer, but the role does not itself grant participant downloads or access to the Cloudflare Account and Worker Secrets. Limit the role to a small number of people and use organizer credentials for routine `push` and `publish` work.

## 2. Initial configuration

Use Davis CLI v0.5.9 or later.

```bash
davis --version
davis admin --help
```

Register a strong Site Admin code as a Worker Secret. Never store it in the repository, a regular environment variable, an issue, or a Pull Request.

```bash
cd web/davis-web
pnpm exec wrangler secret put DAVIS_ADMIN_CODE
pnpm deploy
```

`DAVIS_ADMIN_ACCESS_REVISION` revokes every existing Site Admin session. If the code leaks or administrators change, replace `DAVIS_ADMIN_CODE`, change the revision, and redeploy. Participant and organizer sessions use separate revisions and are unaffected.

To retain the previous participant and organizer codes as one group, configure `DAVIS_LEGACY_GROUP_ID`. Update the participant and organizer access revisions during migration when every existing session should be replaced by a fresh login.

## 3. Site Admin login

Enter the Site Admin code at the interactive prompt so that it does not appear in a command argument or shell history.

```bash
davis admin login <Davis Web URL>
davis admin status
```

The CLI stores a short-lived Site Admin session, not the Site Admin code. Remove the session after administrative work or when using a shared computer.

```bash
davis admin logout
```

## 4. Create an access group

Each access group has exactly one paired participant code and organizer code. Use a stable group ID made of lowercase letters, digits, and hyphens that identifies the organization and period.

```bash
davis admin group-create municipality-a-2026
```

The two codes are displayed only once. Store them immediately in separate secure locations. Never distribute the organizer code to participants.

When the group's organizer first pushes a new unpublished dataset, Davis assigns that dataset to the same group. A routine organizer cannot take over a dataset already assigned to another group.

```bash
davis operator login <Davis Web URL>
davis push <dataset ID>
```

## 5. Change dataset download permissions

A Site Admin can permit multiple groups to download one dataset. `dataset-access` replaces the complete group set; it does not append one group. Include every existing group that should retain access.

```bash
davis admin dataset-access network/matsuyama \
  --group municipality-a-2026 \
  --group research-team-2026
```

Davis checks the current grant not only when issuing a new Download Grant but also when a download uses an already-issued grant. A removed group cannot retrieve the dataset even while its participant session remains valid. Copies already saved on participant devices cannot be technically recalled.

The current CLI requires at least one group. It does not provide a command to hide a dataset from every group, delete a group, or reissue group codes. If a participant code leaks, create a replacement group and replace every affected dataset's grant set without the old group. The old group remains registered but cannot retrieve data when no dataset grants it access. If an organizer code leaks, stop publication and contact the deployment administrator because the old group can still claim a new dataset and publish a catalog; the current CLI cannot fully revoke it. Routine organizers must not edit the private R2 object `access/control.json` manually.

## 6. Store R2 objects with gzip

Starting with v0.5.9, the CLI compresses new and changed objects with gzip on the organizer's device before multipart upload. The Worker stores the encoded object in R2 and serves it unchanged with `Content-Encoding: gzip`. The browser or CLI decodes it on the participant device, so the saved filename and contents remain unchanged.

Because decoded byte positions do not map directly to gzip byte positions, compressed objects do not support Range downloads and are returned as full responses. Legacy uncompressed objects retain Range support.

To migrate existing objects, run the command from a machine whose local content-addressed store contains every object in the current catalog. The default store is `.davis/cache` in the repository. Davis verifies each local object against its BLAKE3 ID and size, compresses it locally, uploads the gzip representation, commits that representation, and then removes the raw R2 copy.

```bash
git status
davis admin storage-compress --yes
```

Specify a non-default store explicitly.

```bash
davis admin storage-compress --yes --store /absolute/path/to/cache
```

The operation is safe to rerun. Already-compressed objects are not uploaded again, and a failed object retains its raw copy. The command reads every catalog object and consumes local CPU and temporary disk space, so verify cache completeness and free space first.

## 7. Move existing codes into a legacy group

To retain `DAVIS_INVITE_CODE` and `DAVIS_OPERATOR_CODE`, configure a legacy group such as `bmss26`.

```text
DAVIS_LEGACY_GROUP_ID=legacy-group-id
```

Migration sequence:

1. Schedule a short maintenance window and record every currently published dataset ID.
2. Configure `DAVIS_LEGACY_GROUP_ID`.
3. Change `DAVIS_ACCESS_REVISION` and `DAVIS_OPERATOR_ACCESS_REVISION`, then deploy the Worker.
4. Sign in as Site Admin and run `davis admin dataset-access <dataset ID> --group <legacy group ID>` for every published dataset.
5. Sign in again with the previous participant and organizer codes and verify downloads and organizer operations.
6. Create new groups and move dataset grants incrementally.

This migration classifies the previous codes as one access group without invalidating the codes themselves. Changing revisions invalidates old sessions, but users can sign in again with the same codes.

## 8. Credential handling and incident response

- Never commit Site Admin, participant, or organizer codes to the repository.
- Never place Site Admin or organizer codes in participant documentation or broadly addressed messages.
- Store codes returned by `group-create` immediately in an organizational password manager; they cannot be retrieved later.
- Change `DAVIS_ADMIN_ACCESS_REVISION` if a Site Admin session leaks.
- Replace both the Secret and revision if the Site Admin code leaks.
- If a participant group code leaks, create a replacement group and replace every affected dataset's grant set without the old group.
- If an organizer group code leaks, stop publication and contact the deployment administrator. Grant migration alone does not fully revoke it because the current CLI cannot delete a group or reissue its codes.
- Site Admin and routine organizer devices do not need R2 credentials.

## 9. Routine checklist

```bash
davis admin status
davis operator status
git status
```

Record the reason, target dataset, and target groups for each permission change and obtain a second review when practical. Use the organizer session, not the Site Admin session, for routine data updates, review, `push`, and `publish`.
