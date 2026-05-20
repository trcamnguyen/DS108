import re

import pandas as pd

EXACT_DROP_TITLES: set[str] = {
    "business architect",
    "sales consultant",
    "it procurement",
    "product marketer",
    "monetization specialist",
    "ad monetization specialist",
    "cad designer",
    "media designer",
    "photo editor",
    "web designer",
    "motion graphics designer",
    "digital artist",
    "illustrator",
    "content moderator",
    "game operator",
    "data operator",
    "maintenance technician",
    "electrical engineer",
    "technical support",
    "designer",
    "game operations",           # normalize về lowercase
    "user acquisition specialist",
    "it communicator",
    "product operations",
    "project coordinator",
    "customer success",
    "it compliance specialist",
    "system analyst",
    "data annotator",
    "cae engineer",
}

DROP_KEYWORDS: list[str] = [
    "graphic",
    "artist",
    "3d", "2d",
    "multimedia",
    "illustrator"  # NOTE: intentional – matches notebook behaviour (missing comma concatenates with next string)
    "media",
    "business development",
    "video editor",
    "animator",
    "marketing",
    "data entry",
    "content",
    "cnc",
    "seo",
    "trainer",
    "writer",
    "plc",
]


def filter_by_standardized_title(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lọc DataFrame dựa trên cột standardized_title (sau LLM).

    Parameters
    ----------
    df : DataFrame có cột 'standardized_title'

    Returns
    -------
    df_keep    : các row được giữ lại
    df_dropped : các row bị loại (để audit)
    """
    title_series = df["standardized_title"].astype(str).str.strip().str.lower()

    keyword_pattern = "|".join(re.escape(kw) for kw in DROP_KEYWORDS)

    mask_keywords = title_series.str.contains(
        keyword_pattern, case=False, na=False, regex=True
    )
    mask_exact_titles = title_series.isin(EXACT_DROP_TITLES)
    mask_sales = (
        title_series.str.contains(r"\bsales\b", case=False, na=False, regex=True)
        & ~title_series.str.contains(
            r"presales|pre-sales|sales engineer", case=False, na=False, regex=True
        )
    )
    mask_account_manager = title_series.eq("account manager")

    drop_mask = mask_keywords | mask_exact_titles | mask_sales | mask_account_manager

    return df[~drop_mask].copy(), df[drop_mask].copy()
