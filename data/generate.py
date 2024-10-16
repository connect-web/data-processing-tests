import random
from typing import List
from data.VectorEntity import VectorEntity
random.seed(42)

SKILLS = ['Overall', 'Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic', 'Cooking', 'Woodcutting', 'Fletching', 'Fishing', 'Firemaking', 'Crafting', 'Smithing', 'Mining', 'Herblore', 'Agility', 'Thieving', 'Slayer', 'Farming', 'Runecraft', 'Hunter', 'Construction']
MINIGAMES = ['Deadman Points', 'League Points', 'Bounty Hunter - Hunter', 'Bounty Hunter - Rogue', 'Bounty Hunter (Legacy) - Hunter', 'Bounty Hunter (Legacy) - Rogue', 'Clue Scrolls (all)', 'Clue Scrolls (beginner)', 'Clue Scrolls (easy)', 'Clue Scrolls (medium)', 'Clue Scrolls (hard)', 'Clue Scrolls (elite)', 'Clue Scrolls (master)', 'LMS - Rank', 'PvP Arena - Rank', 'Soul Wars Zeal', 'Rifts closed', 'Abyssal Sire', 'Alchemical Hydra', 'Artio', 'Barrows Chests', 'Bryophyta', 'Callisto', "Calvar'ion", 'Cerberus', 'Chambers of Xeric', 'Chambers of Xeric: Challenge Mode', 'Chaos Elemental', 'Chaos Fanatic', 'Commander Zilyana', 'Corporeal Beast', 'Crazy Archaeologist', 'Dagannoth Prime', 'Dagannoth Rex', 'Dagannoth Supreme', 'Deranged Archaeologist', 'Duke Sucellus', 'General Graardor', 'Giant Mole', 'Grotesque Guardians', 'Hespori', 'Kalphite Queen', 'King Black Dragon', 'Kraken', "Kree'Arra", "K'ril Tsutsaroth", 'Mimic', 'Nex', 'Nightmare', "Phosani's Nightmare", 'Obor', 'Phantom Muspah', 'Sarachnis', 'Scorpia', 'Skotizo', 'Spindel', 'Tempoross', 'The Gauntlet', 'The Corrupted Gauntlet', 'The Leviathan', 'The Whisperer', 'Theatre of Blood', 'Theatre of Blood: Hard Mode', 'Thermonuclear Smoke Devil', 'Tombs of Amascut', 'Tombs of Amascut: Expert Mode', 'TzKal-Zuk', 'TzTok-Jad', 'Vardorvis', 'Venenatis', "Vet'ion", 'Vorkath', 'Wintertodt', 'Zalcano', 'Zulrah']

SKILLS_COUNT = len(SKILLS)
MINIGAMES_COUNT = len(MINIGAMES)

def generate_sample_dataset(num_samples: int) -> List[VectorEntity]:
    return [create_sample_gpu_player_minigame(i) for i in range(1, num_samples + 1)]


def create_sample_gpu_player_minigame(id: int) -> VectorEntity:
    # Generate random floats between 0 and 1 for skills and minigames
    first_skill_ratios = generate_random_floats(SKILLS_COUNT)
    first_minigames = generate_random_floats(MINIGAMES_COUNT)
    gained_skill_ratios = generate_random_floats(SKILLS_COUNT)
    gained_minigames = generate_random_floats(MINIGAMES_COUNT)

    return VectorEntity(
        id=id,
        skill_ratio_first_scrape=first_skill_ratios,
        minigame_ratio_first_scrape=first_minigames,
        skill_gain_ratio=gained_skill_ratios,
        minigame_gain_ratio=gained_minigames
    )

# Define a function to generate random floats between 0 and 1
def generate_random_floats(size: int):
    return [random.random() for _ in range(size)]





if __name__ == '__main__':
    # Example usage
    sample_dataset = generate_sample_dataset(5)  # Generate 5 sample players

    # Print the generated dataset
    for player in sample_dataset:
        print(player.minigame_ratio_first_scrape)