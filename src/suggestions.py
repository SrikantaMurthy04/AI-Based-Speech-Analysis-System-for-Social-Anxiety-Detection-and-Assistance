def get_suggestion(level):

    if level == "low":
        return "You sound relaxed. Keep it up!"

    elif level == "medium":
        return "Try slowing down your speech and pause between sentences."

    elif level == "high":
        return "Take deep breaths and relax before speaking."

    else:
        return "Unable to determine suggestion."