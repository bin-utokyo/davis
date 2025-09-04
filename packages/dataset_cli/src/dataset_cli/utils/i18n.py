import contextlib
import gettext
import locale
from pathlib import Path
from typing import Literal, cast

from dataset_cli.utils.config import load_user_config

LOCALE_DIR = (Path(__file__).parent.parent / "locales").resolve()
assert LOCALE_DIR.exists(), f"Locale directory does not exist: {LOCALE_DIR}"


def init_translation(language: Literal["ja", "en"] = "ja") -> gettext.NullTranslations:
    """言語コードを指定して翻訳を初期化する"""
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")

    try:
        t = gettext.translation(
            "messages",
            LOCALE_DIR,
            languages=[language],
            fallback=True,
        )

    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load translations for language '{language}': {e}")  # noqa: T201
        return gettext.NullTranslations()

    return t


def get_translator() -> gettext.NullTranslations:
    """ユーザー設定に基づいて翻訳を初期化し、翻訳オブジェクトを返す"""
    user_config = load_user_config()
    lang = cast("Literal['ja', 'en']", user_config.get("lang", "ja"))
    return init_translation(lang)


translator = get_translator()
_ = translator.gettext
