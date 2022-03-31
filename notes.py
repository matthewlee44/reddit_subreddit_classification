import pandas as pd
import numpy as np

def clean_full_data(df):

    # author has entries that are marked "[deleted]" most have no post text, so dropping these
    df = df[df['author'] != "[deleted]"]

    # author_flair_text mostly missing but might have some distinguishing information between class
    # changing missing values to blank string for now ("")
    df['author_flair_text'].replace(np.nan, "", inplace=True)

    # created_utc converting to datetime datatype
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # domain keeping for now
    # id - unique identifier of post; keeping for now
    # link_flair_text keeping for now


    boolean_to_int_cols = [
        "allow_live_comments",
        "author_premium",
        # gildings only a very small amount of non-missing values for both classes (less than .001); dropping column
        "gildings",
        "is_cross_postable",
        # is_created_from_ads_ui 82-85% false, missing values for rest, roughly same proportions between classes
        # dropping column
        "is_created_from_ads_ui",
        "is_reddit_media_domain",
        "is_robot_indexable",
        "is_self",
        "author_is_blocked",
        "is_robot_indexable",
        "is_self",
    ]

    drop_cols = [
        # not enough non-False values
        "is_video",
        # not enough non-missing values
        "author_cakeday",
        "author_flair_background_color",
        "banned_by",
        "call_to_action",
        "category",
        "distinguished",
        "edited",
        "gilded",
        "is_meta",
        "is_original_content",
        # author_flair_css_class only has non-missing value so dropping column
        "author_flair_css_class",
        # author_flair_richtext is only in dating subreddit and more than 99% is missing; dropping column
        "author_flair_richtext",
        # author_flair_type more than 99.5% of column has 1 value; dropping column
        "author_flair_type",
        # author_full_name corresponds exactly with author so dropping this column
        "author_fullname",
        # author_patreon_flair all False values so dropping column
        "author_patreon_flair",    
        # awarders column is all empty; dropping column
        "awarders",
        # can_mod_post all empty; dropping column
        "can_mod_post",
        # contest_mode all empty; dropping column
        "contest_mode",

        # basically same info as other column
        "crosspost_parent_list",
        "link_flair_richtext",
        "link_flair_type",
        # full_link all unique values corresponding with title; 
        # includes name of subreddit in link so may make classification too simple; dropping column
        "full_link",

    ]

    # all_awardings setting to 1 if present, 0 of blank ([])
    df['all_awardings'] = df['all_awardings'].apply(lambda x: 0 if x == "[]" else 1)
    
    # Changing to 0/1 based on whether missing value or not
    df['crosspost_parent_list'] = df['crosspost_parent_list'].apply(lambda x: 0 if x == np.nan else 1)
    df['author_flair_template_id'] = df['author_flair_template_id'].apply(lambda x: 0 if x == np.nan else 1)
    df['author_flair_text_color'] = df['author_flair_text_color'].apply(lambda x: 0 if x == np.nan else 1)
    df['crosspost_parent'] = df['crosspost_parent'].apply(lambda x: 0 if x == np.nan else 1)
    df['is_reddit_media_domain'] = df['is_reddit_media_domain'].apply(lambda x: 0 if x == np.nan else 1)
    df['link_flair_background_color'] = df['link_flair_background_color'].apply(lambda x: 0 if x == np.nan else 1)
    df['link_flair_css_class'] = df['link_flair_css_class'].apply(lambda x: 0 if x == np.nan else 1)
    df['link_flair_template_id'] = df['link_flair_template_id'].apply(lambda x: 0 if x == np.nan else 1)


