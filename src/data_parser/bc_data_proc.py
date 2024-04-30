
import os
import json
from src.utils import DATA_DIR

if __name__ == '__main__':
    
    # File path
    file_name = "games.json"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # load file as string
    with open(file_path, "r") as f:
        games_str = f.read()

    # replace a single quote with a double quote
    games_str = games_str.replace("'", '"')

    # replace None with null
    games_str = games_str.replace("None", "null")

    # replace True with true
    games_str = games_str.replace("True", "true")

    # replace False with false
    games_str = games_str.replace("False", "false")

    # write the string to a new file
    temp_file_name = file_name.split(".")[0] + "_temp.json"
    out_file = os.path.join(DATA_DIR, temp_file_name)
    with open(out_file, "w") as f:
        f.write(games_str)

    # load file as json
    file_path = os.path.join(DATA_DIR, temp_file_name)
    with open(file_path, "r") as f:
        games = json.load(f)
    print(f"Found {len(games)} games in {file_name}")
    
    # filter out only games with 5 players and roles Merlin and Assassin
    def should_include(game):
        if not game["gameMode"] == "avalon":
            return False
        if not game["numberOfPlayers"] == 5:
            return False
        if len(game["cards"]) != 0:
            return False
        if len(game["numFailsHistory"]) == 0:
            return False
        if game["howTheGameWasWon"] == "Hammer rejected.":
            return False
        if len(game["roles"]) < 2: # Require Merlin and Assassin
            return False
        for role in game["roles"]:
            if role not in ["Merlin", "Assassin"]:
                return False
        return True
    filtered_games = [game for game in games if should_include(game)]
    
    # write filtered games to a new file
    final_file_name = file_name.split(".")[0] + "_final.json"
    print(f"Writing {len(filtered_games)} filtered games to {final_file_name}")
    with open(os.path.join(DATA_DIR, final_file_name), "w") as f:
        json.dump(filtered_games, f, indent=4)