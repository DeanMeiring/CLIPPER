"""Evolution chains for Ball Evolution: what drops, and what it grows into.

Each theme is a chain of 8-9 emoji, small to big, with its own hook line(s)
and a word for the counter ("312 drops dropped"). Emoji art is the bundled
Fluent Emoji 3D set (MIT, clipper/assets/emoji); ``code_for`` finds an
emoji's file. Hooks are plain text (the video font has no emoji glyphs).

Dean asked for "at least 50 different" themes so a twice-a-day channel
rotates a lot; ``pick_recipe`` keeps recently used ones off.
"""
from __future__ import annotations

from pathlib import Path

EMOJI_DIR = Path(__file__).parent / "assets" / "emoji"

# key: (hooks, counter word, "emoji Name, emoji Name, ...")
_SPECS = {
    "animals": (["Can a bee become the last animal?", "Can it make the last one?"], "bees",
                "🐝 Bee, 🐀 Mouse, 🐸 Frog, 🐔 Chicken, 🐱 Cat, 🐶 Dog, 🐼 Panda, 🦁 Lion, 🦄 Unicorn"),
    "sports": (["Can a baseball win the trophy?", "Will it reach the last ball?"], "baseballs",
               "⚾ Baseball, 🎾 Tennis ball, 🏐 Volleyball, ⚽ Football, 🏀 Basketball, 🏈 Rugby ball, 🎳 Bowling, 🥇 Gold medal, 🏆 Trophy"),
    "food": (["Can cookies become the final food?", "What's the last food?"], "cookies",
             "🍪 Cookie, 🍩 Donut, 🍿 Popcorn, 🍟 Fries, 🌭 Hot dog, 🍔 Burger, 🍕 Pizza, 🎂 Cake"),
    "space": (["Can stardust make the last one?", "What's at the end of space?"], "sparks",
              "✨ Stardust, ⭐ Star, 🌟 Bright star, 🌙 Moon, 🌍 Earth, ☀️ Sun, 🚀 Rocket, 🛸 UFO, 👽 Alien"),
    "money": (["Can one coin make you rich?", "Coin to... what?"], "coins",
              "🪙 Coin, 💵 Cash, 💸 Flying cash, 💳 Card, 💰 Money bag, 💎 Diamond, 👑 Crown, 🤑 Rich"),
    "laughs": (["Can a meh face become the happiest?", "How happy can it get?"], "faces",
               "😐 Meh, 🙂 Smile, 😊 Happy, 😄 Grin, 😆 Laugh, 😂 Crying laughing, 🤣 Rolling, 🤩 Starstruck, 🥳 Party"),
    "vehicles": (["Can a car become the fastest thing?", "Car to... what?"], "cars",
                 "🚗 Car, 🚓 Police car, 🚌 Bus, 🚂 Train, 🏎️ Race car, ✈️ Plane, 🚀 Rocket, 🛸 UFO"),
    "weather": (["Can one drop make the sun?", "What's after the storm?"], "drops",
                "💧 Drop, 💦 Splash, 🌧️ Rain, ⛈️ Storm, ⚡ Lightning, 🌪️ Tornado, 🌊 Wave, 🌈 Rainbow, ☀️ Sun"),
    "ocean": (["Can a shrimp become the king of the sea?", "What's the biggest in the ocean?"], "shrimp",
              "🦐 Shrimp, 🐟 Fish, 🐠 Tropical fish, 🐡 Blowfish, 🦀 Crab, 🐙 Octopus, 🐬 Dolphin, 🦈 Shark, 🐋 Whale"),
    "bugs": (["Can a germ become a dragon?", "What does an ant grow into?"], "germs",
             "🦠 Germ, 🐜 Ant, 🐞 Ladybug, 🐛 Caterpillar, 🦗 Cricket, 🕷️ Spider, 🦂 Scorpion, 🦋 Butterfly, 🐉 Dragon"),
    "farm": (["Can a chick run the whole farm?", "What's the biggest on the farm?"], "chicks",
             "🐣 Hatchling, 🐥 Chick, 🐔 Hen, 🦆 Duck, 🐑 Sheep, 🐖 Pig, 🐐 Goat, 🐄 Cow, 🐎 Horse"),
    "life": (["Can a germ evolve into a person?", "Germ to human in one minute?"], "germs",
             "🦠 Germ, 🐟 Fish, 🐸 Frog, 🦎 Lizard, 🦖 T-rex, 🐒 Monkey, 🦍 Gorilla, 🧑 Human, 🧙 Wizard"),
    "fruit": (["Can a blueberry become a watermelon?", "What's the biggest fruit?"], "berries",
              "🫐 Blueberry, 🍒 Cherries, 🍓 Strawberry, 🍋 Lemon, 🍎 Apple, 🍑 Peach, 🥭 Mango, 🍍 Pineapple, 🍉 Watermelon"),
    "veggies": (["Can a bean become a pumpkin?", "Which vegetable wins?"], "beans",
                "🫘 Beans, 🧄 Garlic, 🧅 Onion, 🥕 Carrot, 🌶️ Chili, 🥦 Broccoli, 🍆 Eggplant, 🥬 Lettuce, 🎃 Pumpkin"),
    "sweets": (["Can a candy become the best dessert?", "What's the final dessert?"], "candies",
               "🍬 Candy, 🍭 Lollipop, 🍫 Chocolate, 🧁 Cupcake, 🍩 Donut, 🍦 Ice cream, 🍰 Cake slice, 🥧 Pie, 🎂 Birthday cake"),
    "drinks": (["Can one drop become the best drink?", "Which drink wins?"], "drops",
               "💧 Drop, 🧊 Ice, 🥛 Milk, ☕ Coffee, 🍵 Tea, 🧃 Juice box, 🥤 Soda, 🧋 Bubble tea, 🍹 Smoothie"),
    "plants": (["Can a sprout become a giant tree?", "What does a seed grow into?"], "sprouts",
               "🌱 Sprout, 🌿 Herb, 🍀 Clover, 🌷 Tulip, 🌹 Rose, 🌻 Sunflower, 🌵 Cactus, 🌴 Palm, 🌳 Tree"),
    "flowers": (["Can a daisy become a bouquet?", "What's the last flower?"], "daisies",
                "🌼 Daisy, 🌸 Blossom, 🌺 Hibiscus, 🌷 Tulip, 🌹 Rose, 🪷 Lotus, 🌻 Sunflower, 💐 Bouquet"),
    "moon": (["Can a new moon become a galaxy?", "What comes after the full moon?"], "moons",
             "🌑 New moon, 🌒 Crescent, 🌓 Half moon, 🌔 Gibbous, 🌕 Full moon, 🌍 Earth, ☀️ Sun, 🌟 Star, 🌌 Galaxy"),
    "treasure": (["Can a rock become a crown?", "Rock to riches?"], "rocks",
                 "🪨 Rock, 🥉 Bronze, 🥈 Silver, 🥇 Gold, 💍 Ring, 💎 Diamond, 👑 Crown, 🏆 Trophy"),
    "music": (["Can one note become a concert?", "Which instrument wins?"], "notes",
              "🎵 Note, 🎶 Notes, 🥁 Drum, 🪇 Maracas, 🎸 Guitar, 🎹 Piano, 🎺 Trumpet, 🎷 Sax, 🎤 Mic"),
    "tech": (["Can a floppy disk become a rocket?", "Old tech to the future?"], "disks",
             "💾 Floppy, 📼 Tape, 📟 Pager, 📱 Phone, 💻 Laptop, 🖥️ Computer, 🤖 Robot, 🛰️ Satellite, 🚀 Rocket"),
    "buildings": (["Can a tent become a city?", "What does a hut grow into?"], "tents",
                  "⛺ Tent, 🛖 Hut, 🏠 House, 🏡 Home, 🏢 Office, 🏬 Mall, 🏰 Castle, 🏯 Palace, 🏙️ City"),
    "boats": (["Can a canoe become a cruise ship?", "What's the biggest boat?"], "canoes",
              "🛶 Canoe, ⛵ Sailboat, 🚤 Speedboat, 🛥️ Yacht, ⛴️ Ferry, 🚢 Ship, 🛳️ Cruise ship, 🏝️ Island"),
    "sky": (["Can a kite reach space?", "How high can it go?"], "kites",
            "🪁 Kite, 🎈 Balloon, 🪂 Parachute, 🛩️ Small plane, ✈️ Jet, 🚁 Helicopter, 🚀 Rocket, 🛸 UFO, 🌌 Galaxy"),
    "fire": (["Can a spark become the sun?", "How big can the fire get?"], "sparks",
             "✨ Spark, 🔥 Fire, 💥 Boom, 🧨 Dynamite, ☄️ Comet, 🌋 Volcano, 🌞 Sun, 💫 Supernova"),
    "rage": (["How angry can it get?", "Calm to... what?"], "calm faces",
             "🙂 Calm, 😐 Meh, 😑 Annoyed, 😒 Unamused, 😤 Huffing, 😠 Angry, 😡 Furious, 🤬 Rage, 🤯 Mind blown"),
    "cool": (["Can a smile become the coolest?", "How cool can it get?"], "smiles",
             "😶 Quiet, 🙂 Smile, 😊 Happy, 😁 Grin, 😎 Cool, 🤠 Cowboy, 🤑 Rich, 🥳 Party, 👑 King"),
    "spooky": (["Can a pumpkin become the scariest?", "What's the scariest?"], "pumpkins",
               "🎃 Pumpkin, 👻 Ghost, 🦇 Bat, 🕷️ Spider, 💀 Skull, 🧟 Zombie, 🧛 Vampire, 🧙 Witch, 😈 Devil"),
    "winter": (["Can a snowflake become Santa?", "What's the last thing in winter?"], "snowflakes",
               "❄️ Snowflake, ⛄ Snowman, 🧣 Scarf, 🎄 Tree, 🎁 Gift, 🔔 Bell, 🦌 Reindeer, 🛷 Sled, 🎅 Santa"),
    "party": (["Can a balloon start the biggest party?", "How big does the party get?"], "balloons",
              "🎈 Balloon, 🎉 Popper, 🎊 Confetti, 🎂 Cake, 🎁 Gift, 🪅 Pinata, 🎆 Fireworks, 🥳 Party"),
    "pets": (["Can a goldfish become the best pet?", "Which pet wins?"], "goldfish",
             "🐠 Goldfish, 🐢 Turtle, 🐹 Hamster, 🐰 Bunny, 🐈 Cat, 🐕 Dog, 🦮 Guide dog, 🐴 Pony"),
    "jungle": (["Can a monkey rule the jungle?", "Who's the king of the jungle?"], "monkeys",
               "🐒 Monkey, 🦥 Sloth, 🦜 Parrot, 🐍 Snake, 🐆 Leopard, 🐅 Tiger, 🦁 Lion, 🦏 Rhino, 🐘 Elephant"),
    "arctic": (["Can an ice cube become a whale?", "What's the biggest in the snow?"], "ice cubes",
               "🧊 Ice, 🐧 Penguin, 🦭 Seal, 🦊 Fox, 🐺 Wolf, 🦌 Deer, 🐻‍❄️ Polar bear, 🐋 Whale"),
    "birds": (["Can a chick become an eagle?", "Which bird wins?"], "chicks",
              "🐤 Chick, 🐦 Bird, 🕊️ Dove, 🦆 Duck, 🦉 Owl, 🦜 Parrot, 🦩 Flamingo, 🦚 Peacock, 🦅 Eagle"),
    "balls": (["Can a ping pong ball become the Earth?", "What's the biggest ball?"], "balls",
              "🏓 Ping pong, 🎱 Pool ball, ⚾ Baseball, 🥎 Softball, 🏐 Volleyball, ⚽ Football, 🏀 Basketball, 🌍 Earth"),
    "tools": (["Can a paperclip become a crane?", "What's the biggest tool?"], "paperclips",
              "📎 Paperclip, 🔩 Bolt, 🔧 Wrench, 🔨 Hammer, 🪚 Saw, ⚙️ Gear, 🛠️ Toolkit, 🚜 Tractor, 🏗️ Crane"),
    "school": (["Can a pencil earn a degree?", "Pencil to graduate?"], "pencils",
               "✏️ Pencil, 🖍️ Crayon, 📏 Ruler, 📒 Notebook, 📚 Books, 🎒 Backpack, 🧪 Science, 🔬 Microscope, 🎓 Graduate"),
    "fashion": (["Can a sock become a crown?", "What's the last outfit?"], "socks",
                "🧦 Sock, 🧤 Gloves, 🧢 Cap, 👕 Shirt, 👖 Jeans, 👗 Dress, 🧥 Coat, 👠 Heels, 👑 Crown"),
    "games": (["Can a dice become the champion?", "Which game wins?"], "dice",
              "🎲 Dice, 🧩 Puzzle, ♟️ Pawn, 🃏 Joker, 🎯 Target, 🎳 Bowling, 🎮 Controller, 🕹️ Joystick, 🏆 Trophy"),
    "hearts": (["Can a white heart become love?", "Which heart is last?"], "hearts",
               "🤍 White, 💛 Yellow, 💚 Green, 💙 Blue, 💜 Purple, ❤️ Red, 💗 Growing, 💖 Sparkle, 💝 Gift"),
    "colors": (["Can white become a rainbow?", "Which colour wins?"], "dots",
               "⚪ White, 🟡 Yellow, 🟠 Orange, 🔴 Red, 🟣 Purple, 🔵 Blue, 🟢 Green, 🟤 Brown, ⚫ Black"),
    "numbers": (["Can 1 make it all the way to 9?", "Count with the balls!"], "ones",
                "1️⃣ One, 2️⃣ Two, 3️⃣ Three, 4️⃣ Four, 5️⃣ Five, 6️⃣ Six, 7️⃣ Seven, 8️⃣ Eight, 9️⃣ Nine"),
    "people": (["Can a baby become a wizard?", "Baby to... what?"], "babies",
               "👶 Baby, 🧒 Kid, 👦 Boy, 🧑 Adult, 🧔 Beard, 👴 Grandpa, 🧓 Elder, 🧙 Wizard, 👑 King"),
    "heroes": (["Can a kid become a superhero?", "Who's the strongest?"], "kids",
               "🧒 Kid, 🧑‍🎓 Student, 👷 Builder, 👮 Police, 🧑‍🚒 Firefighter, 🧑‍🚀 Astronaut, 🥷 Ninja, 🦸 Superhero"),
    "reptiles": (["Can a lizard become a dragon?", "What does a lizard grow into?"], "lizards",
                 "🦎 Lizard, 🐢 Turtle, 🐍 Snake, 🐊 Croc, 🦕 Dino, 🦖 T-rex, 🐲 Dragon face, 🐉 Dragon"),
    "breakfast": (["Can an egg become the best breakfast?", "What's the last breakfast?"], "eggs",
                  "🥚 Egg, 🍳 Fried egg, 🥓 Bacon, 🥞 Pancakes, 🧇 Waffle, 🥐 Croissant, 🥯 Bagel, 🍞 Bread, 🥪 Sandwich"),
    "asian": (["Can a rice ball become a feast?", "What's the final dish?"], "rice balls",
              "🍙 Rice ball, 🍘 Cracker, 🍣 Sushi, 🍤 Shrimp, 🥟 Dumpling, 🍜 Ramen, 🍱 Bento, 🥡 Takeout, 🍲 Hot pot"),
    "nature": (["Can a rock become a volcano?", "What's the biggest in nature?"], "rocks",
               "🪨 Rock, 🌾 Grass, 🌲 Pine, ⛰️ Hill, 🏔️ Mountain, 🌋 Volcano, 🏝️ Island, 🌍 Earth, 🌌 Galaxy"),
    "wheels": (["Can a scooter become a truck?", "What's the biggest on wheels?"], "scooters",
               "🛴 Scooter, 🚲 Bike, 🛵 Moped, 🏍️ Motorbike, 🚗 Car, 🚙 SUV, 🛻 Pickup, 🚚 Truck, 🚛 Big rig"),
    "trains": (["Can a tram become a bullet train?", "Which train is fastest?"], "trams",
               "🚋 Tram, 🚃 Carriage, 🚞 Mountain train, 🚂 Steam train, 🚆 Train, 🚄 Fast train, 🚅 Bullet train, 🚝 Monorail"),
    "robots": (["Can a pixel alien become a UFO?", "What's the final robot?"], "aliens",
               "👾 Pixel alien, 🤖 Robot, 👽 Alien, 🛸 UFO, 🛰️ Satellite, 🪐 Planet, 🌌 Galaxy, 🌠 Shooting star"),
    "wild": (["Can a mouse become an elephant?", "Who's the biggest?"], "mice",
             "🐭 Mouse, 🐿️ Chipmunk, 🐇 Rabbit, 🦊 Fox, 🐺 Wolf, 🐻 Bear, 🦬 Bison, 🦛 Hippo, 🐘 Elephant"),
    "kitchen": (["Can a spoon cook a feast?", "What's the last thing in the kitchen?"], "spoons",
                "🥄 Spoon, 🍴 Fork, 🔪 Knife, 🥢 Chopsticks, 🍳 Pan, 🫕 Pot, 🍲 Stew, 🥘 Paella, 🎂 Cake"),
    "time": (["Can a second become a year?", "What's the longest time?"], "seconds",
             "⏱️ Stopwatch, ⏳ Hourglass, ⏰ Alarm, 🕰️ Clock, 📅 Calendar, 🗓️ Planner, 🌍 Earth, ☀️ Sun"),
    "magic": (["Can a sparkle become pure magic?", "What's the strongest magic?"], "sparkles",
              "✨ Sparkle, 🪄 Wand, 🔮 Crystal ball, 🧪 Potion, 📜 Scroll, 🧞 Genie, 🧚 Fairy, 🧙 Wizard, 🦄 Unicorn"),
    "gym": (["Can a pushup become a champion?", "Weak to... what?"], "pushups",
            "🧘 Stretch, 🤸 Flip, 🏃 Run, 🚴 Cycle, 🏋️ Lift, 🤼 Wrestle, 🥊 Boxing, 🥇 Gold, 🏆 Champion"),
    "camping": (["Can a match light up the night?", "What's the last thing at camp?"], "matches",
                "🔥 Fire, 🪵 Log, 🔦 Flashlight, 🎒 Backpack, 🏕️ Campsite, 🌲 Forest, 🏔️ Mountain, 🌌 Night sky"),
}


def _parse(spec: str) -> list:
    out = []
    for item in spec.split(", "):
        emoji, name = item.split(" ", 1)
        out.append((emoji, name))
    return out


def code_for(emoji: str) -> str | None:
    """The emoji's file name (without .webp) in assets/emoji, if it's there."""
    plain = emoji.replace("️", "")
    tries = ["-".join(f"{ord(c):x}" for c in emoji), "-".join(f"{ord(c):x}" for c in plain),
             "-".join(f"{ord(c):x}" for c in plain) + "-fe0f"]
    for t in tries:
        if (EMOJI_DIR / f"{t}.webp").is_file():
            return t
    return None


def build() -> dict:
    themes = {}
    for key, (hooks, unit, spec) in _SPECS.items():
        chain = []
        for emoji, name in _parse(spec):
            code = code_for(emoji)
            if code:
                chain.append((code, name))
        if len(chain) >= 7:          # a theme with missing art still needs a real chain
            themes[key] = {"chain": chain, "unit": unit, "hooks": hooks}
    return themes
