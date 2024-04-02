from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
import random


def create_notifications_data(filepaths: list[str]):
    # Get only "Notification" column of all files and concat in one dataframe
    dataframes = [pd.read_csv(filepath, sep=';')["Notification"] for filepath in filepaths]
    notifications = pd.DataFrame(pd.concat(dataframes, ignore_index=True))
    notifications = notifications.dropna(ignore_index=True)

    # Shuffle data
    shuffled_notifications = notifications.sample(frac=1, ignore_index=True)

    # Create more notifications with common errors to ensure quality of models
    notifs_with_evaluation_errors = shuffled_notifications.iloc[:800, :].copy()
    notifs_with_duplication_errors = shuffled_notifications.iloc[800:1700, :].copy()
    notifs_with_typos = shuffled_notifications.iloc[1700:2000, :].copy()
    notifs_with_grammar_errors = shuffled_notifications.iloc[2000:, :].copy()

    create_samples_with_evaluation_errors(notifs_with_evaluation_errors)
    create_samples_with_duplication_errors(notifs_with_duplication_errors)
    create_samples_with_typos(notifs_with_typos)
    create_samples_with_grammar_errors(notifs_with_grammar_errors)

    dataframes_to_concat = [shuffled_notifications, notifs_with_evaluation_errors, notifs_with_duplication_errors,
                            notifs_with_typos, notifs_with_grammar_errors]
    full_notifications = pd.concat(dataframes_to_concat, ignore_index=True)

    # Mark generated notifications as errors ("Target" = 1)
    full_notifications.insert(1, "Target", np.array([0] * len(shuffled_notifications) +
                              [1] * len(shuffled_notifications)), allow_duplicates=True)
    full_notifications = full_notifications.sample(frac=1, ignore_index=True)

    full_notifications.to_csv("../examples/notifications.csv", sep=':', index=False)

    # Use as following:
    # notifs = pd.read_csv("notifications.csv", sep=':')


def create_samples_with_evaluation_errors(sample_data: pd.DataFrame):
    """
    Imitate inappropriate appearance of pattern in notification.
    """

    pattern = '${(word)|numberformat("0.##")}'

    for i in range(len(sample_data)):
        word_to_replace = random.choice(sample_data["Notification"][i].split(' '))
        pattern_with_word = pattern.replace('word', word_to_replace)
        notification_with_error = sample_data["Notification"][i].replace(word_to_replace, pattern_with_word)
        sample_data["Notification"][i] = notification_with_error


def create_samples_with_duplication_errors(sample_data: pd.DataFrame, start_index=800):
    """
    Duplicate random word in notification.
    """

    for i in range(start_index, start_index + len(sample_data)):
        word_to_duplicate = random.choice(sample_data["Notification"][i].split(' '))
        notification_with_error = sample_data["Notification"][i].replace(word_to_duplicate,
                                                                         word_to_duplicate + " " + word_to_duplicate)
        sample_data["Notification"][i] = notification_with_error


def create_samples_with_typos(sample_data: pd.DataFrame, start_index=1700):
    """
    Imitate typos in random word using letters duplication.
    """

    for i in range(start_index, start_index + len(sample_data)):
        word_to_typo = random.choice(sample_data["Notification"][i].split(' '))
        letter_to_fail_idx = random.randint(0, len(word_to_typo) - 1)
        times_to_fail = random.randint(2, len(word_to_typo) * 2)
        typo = word_to_typo.replace(word_to_typo[letter_to_fail_idx], word_to_typo[letter_to_fail_idx] * times_to_fail)
        notification_with_error = sample_data["Notification"][i].replace(word_to_typo, typo)
        sample_data["Notification"][i] = notification_with_error


def create_samples_with_grammar_errors(sample_data: pd.DataFrame, start_index=2000):
    """
    Change form of random word in notification.
    """

    stemmer = SnowballStemmer("russian")
    for i in range(start_index, start_index + len(sample_data)):
        random_word = random.choice(sample_data["Notification"][i].split(' '))
        word_to_fail = random_word
        count = 0
        while word_to_fail == random_word and count < 5:
            count += 1
            word_to_fail = stemmer.stem(word_to_fail)
        if count >= 5:
            word_to_fail = word_to_fail.replace(word_to_fail[-1], word_to_fail[-1] * 2)
        notification_with_error = sample_data["Notification"][i].replace(random_word, word_to_fail)
        sample_data["Notification"][i] = notification_with_error
