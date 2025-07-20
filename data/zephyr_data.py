"""
Sample social media data about Zephyr Innovations products.
"""

import datetime
from random import randint


# Generate dates for the last 90 days
def generate_date():
    today = datetime.datetime.now()
    days_ago = randint(1, 90)
    return (today - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")


# List of commenters (X.com-style handles)
COMMENTERS = [
    "WalleyCoyote",  # Main character, frequent poster
    "DesertRunner42",  # Another coyote in the chase
    "CactusCrashAZ",  # Arizona-based mischief-maker
    "BeepBeepBuster",  # Randy Runner nemesis
    "CanyonCoyote7",  # Desert dweller
    "TumbleweedTrapper",  # Trap enthusiast
]

# Sample social media posts about Zephyr Innovations products
SOCIAL_MEDIA_POSTS = [
    {
        "id": 1,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Just got my Rocket Skates from Zephyr Innovations! Zoomed past that Randy Runner in seconds!",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 2,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Zephyr’s Rocket Skates are a disaster! Crashed into a cactus at 60 mph. Zero stars.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 3,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "The Giant Magnet worked perfectly—pulled that pesky Randy Runner right off the cliff!",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 4,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "Zephyr’s TNT Sticks are explosive fun! Blew up a canyon wall with one try.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 5,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Tried the Anvil Drop Kit today. Missed that Randy Runner and flattened my tail instead.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 6,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Zephyr’s Bird Seed lured that Randy Runner, but it ate faster than I could catch it.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 7,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are fast, I guess. Steering’s tricky, but I’ll get the hang of it.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 8,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "This Giant Magnet is unreal! Dragged my anvil across the desert in seconds.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 9,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "WARNING: Zephyr TNT Sticks explode too fast! I’m still picking soot out of my fur.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 10,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit is solid—literally. Dropped it on my foot setting it up.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 11,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "The Bird Seed works fine. Randy Runner loved it, but I’m still hungry.",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 12,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Five stars for Zephyr’s Rocket Skates! Chased that Randy Runner off a cliff!",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 13,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Giant Magnet failed me. Attracted a train instead of that Randy Runner.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 14,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks are decent. Blew up half the desert, but that Randy Runner still got away.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 15,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Love the Anvil Drop Kit! Perfect weight for smashing boulders.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 16,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Zephyr Bird Seed is a scam. Randy Runner ate it all and laughed at me.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 17,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are wild! Went so fast I overshot the canyon.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 18,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Neutral on the Giant Magnet. It works, but my plan didn’t.",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 19,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "Best TNT Sticks ever! Made a crater big enough to trap anyone—except that Randy Runner.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 20,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit instructions are awful. Dropped it on my head twice.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 21,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are a blast! Literally. Singed my tail but almost caught that Randy Runner.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 22,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Zephyr’s Giant Magnet pulled my TNT right out of my paws. Terrible design.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 23,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks went off too soon again. I’m typing this from a cliff bottom.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 24,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit is genius! Timed it perfectly, but that Randy Runner dodged it.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 25,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Bird Seed from Zephyr is top-notch. Too bad it attracts ravens too.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 26,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Caught a glimpse of that Randy Runner with these Rocket Skates before hitting a rock.",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 27,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Giant Magnet is so strong it yanked my watch off. Impressive but annoying.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 28,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "Zephyr TNT Sticks are reliable—blew up exactly where I didn’t want them to.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 29,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit saved the day! Squashed a boulder blocking my path.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 30,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "This Bird Seed is okay. Randy Runner ate it, but I tripped over the bag.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 31,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are too fast! Ended up in the next state chasing shadows.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 32,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Love this Giant Magnet! Caught a falling anvil mid-air.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 33,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks fizzled out halfway. Zephyr needs better quality control.",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 34,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit is a dud. Anvil didn’t drop—just hung there mocking me.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 35,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Zephyr’s Bird Seed worked! Randy Runner stopped to eat, then outran me anyway.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 36,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are thrilling! Crashed spectacularly but I’ll try again.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 37,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Giant Magnet attracted my lunch instead of the target. Mixed feelings.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 38,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks are awesome! Made a boom that echoed for miles.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 39,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit broke on first use. Zephyr owes me a refund.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 40,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Bird Seed is premium quality. Randy Runner can’t resist it!",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 41,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates need brakes! Flew off a cliff and I’m still falling.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 42,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Giant Magnet is too powerful—stuck to my fridge for hours.",
        "commenter": "CanyonCoyote7",
    },
    {
        "id": 43,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks are unpredictable. Blew up my plan and my dignity.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 44,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit is flawless! Dropped right where I aimed—missed that Randy Runner though.",
        "commenter": "TumbleweedTrapper",
    },
    {
        "id": 45,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Bird Seed spilled everywhere. Randy Runner ate it off my paws.",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 46,
        "product": "Rocket Skates",
        "date": generate_date(),
        "text": "Rocket Skates are a game-changer! Almost had that Randy Runner in my grasp.",
        "commenter": "BeepBeepBuster",
    },
    {
        "id": 47,
        "product": "Giant Magnet",
        "date": generate_date(),
        "text": "Giant Magnet malfunctioned and stuck to my TNT. Disaster ensued.",
        "commenter": "CactusCrashAZ",
    },
    {
        "id": 48,
        "product": "TNT Sticks",
        "date": generate_date(),
        "text": "TNT Sticks from Zephyr are top-tier. Blew up the perfect trap!",
        "commenter": "WalleyCoyote",
    },
    {
        "id": 49,
        "product": "Anvil Drop Kit",
        "date": generate_date(),
        "text": "Anvil Drop Kit is heavy duty. Nearly caught that speedy Randy Runner.",
        "commenter": "DesertRunner42",
    },
    {
        "id": 50,
        "product": "Bird Seed",
        "date": generate_date(),
        "text": "Bird Seed is useless. Randy Runner ate it and beeped in my face.",
        "commenter": "TumbleweedTrapper",
    },
]

# Sort by date to simulate a real-world dataset
SOCIAL_MEDIA_POSTS.sort(key=lambda x: x["date"])
