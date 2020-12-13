# todo This file should be used to store common serialisations across aggregating data

def createPostDict(date, post_text, likes, comments, shares, source ="self"):
    '''
    Creates a dictionary containing the pertinent information from a social media post. This should later be added to a list
    of other posts from that account and then added to a master dictionary.
    :param date:
    :param post_text:
    :param likes:
    :param comments:
    :param shares:
    :param source:
    :return: a dictionary containing pertinent post information
    '''
    return {"text":post_text, "likes":likes, "comments":comments, "shares":shares, "source":source, "date":date}

def createWholeUserDict(unique_id, reddit_list, instagram_list,twitter_list, survey_data):
    return {"id":unique_id,"reddit":reddit_list,"instagram":instagram_list,"twitter":twitter_list, "survey":survey_data}