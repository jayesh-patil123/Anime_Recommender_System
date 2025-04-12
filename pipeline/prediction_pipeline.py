from config.paths_config import *
from utils.helpers import *
from src.logger import setup_logger as get_logger

# Initialize logger
logger = get_logger(__name__)

def hybrid_recommendation(user_id, user_weight=0.5, content_weight=0.5):
    """
    Hybrid recommendation system for personalized anime recommendations.
    Combines content-based filtering and user-based collaborative filtering.
    """
    try:
        logger.info(f"[START] Hybrid recommendation pipeline for user_id: {user_id}")

        # Step 1: Fetch user preferences
        user_pref = get_user_preferences(user_id, RATING_DF, DF)
        if user_pref.empty:
            logger.warning(f"[WARNING] No preferences found for user_id: {user_id}. Falling back to random recommendations.")
            return get_random_anime(n=10)

        # Step 2: Find similar users
        logger.info(f"[INFO] Finding similar users for user_id: {user_id}")
        similar_users = find_similar_users(user_id, USER_WEIGHTS_PATH, USER2USER_ENCODED, USER2USER_DECODED)
        if similar_users.empty:
            logger.warning(f"[WARNING] No similar users found for user_id: {user_id}. Using fallback recommendations.")
            return get_random_anime(n=10)

        # Step 3: Generate user-based recommendations
        logger.info(f"[INFO] Getting user-based recommendations.")
        user_recommended_animes = get_user_recommendations(similar_users, user_pref, DF, SYNOPSIS_DF, RATING_DF)
        user_recommended_anime_list = user_recommended_animes["anime_name"].tolist()

        if not user_recommended_anime_list:
            logger.warning(f"[WARNING] No user-based recommendations found for user_id: {user_id}")
            return get_random_anime(n=10)

        # Step 4: Content-based recommendations
        logger.info(f"[INFO] Getting content-based recommendations from user-based anime list.")
        content_recommended_animes = []

        for anime in user_recommended_anime_list:
            logger.debug(f"Finding similar animes for: {anime}")
            similar_animes = find_similar_animes(anime, ANIME_WEIGHTS_PATH, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED, DF)
            if similar_animes is not None and not similar_animes.empty:
                content_recommended_animes.extend(similar_animes["name"].tolist())
            else:
                logger.debug(f"No similar animes found for: {anime}")

        # Step 5: Combine recommendations using weights
        logger.info(f"[INFO] Combining user-based and content-based recommendations.")
        combined_scores = {}

        for anime in user_recommended_anime_list:
            combined_scores[anime] = combined_scores.get(anime, 0) + user_weight

        for anime in content_recommended_animes:
            combined_scores[anime] = combined_scores.get(anime, 0) + content_weight

        # Step 6: Sort and return top 10
        sorted_animes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [anime for anime, _ in sorted_animes[:10]]

        logger.info(f"[COMPLETE] Top 10 recommendations for user_id {user_id}: {top_recommendations}")
        return top_recommendations

    except Exception as e:
        logger.exception(f"[ERROR] Exception in hybrid recommendation for user_id {user_id}: {str(e)}")
        return []
