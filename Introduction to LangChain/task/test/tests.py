from hstest import StageTest, TestedProgram, CheckResult, dynamic_test
import re
import random

class LangchainTest(StageTest):
    # Key keywords for each planet
    test_data = [
        ("Mercury", r"smallest planet|closest to the Sun|elliptical orbit|thin atmosphere|cratered surface|temperature variations"),
        ("Venus", r"second planet|sister planet|thick atmosphere|carbon dioxide|extreme temperatures|retrograde rotation"),
        ("Neptune", r"eighth planet|blue color|methane|Great Dark Spot|gas giant|Triton"),
        ("Pluto", r"dwarf planet|Kuiper Belt|rock|ice|thin atmosphere|five moons|elliptical orbit")
    ]

    @dynamic_test(time_limit=0)
    def test(self):
        # Randomly select aplanet and its associated keyword pattern
        for i in range(4):
            planet_name, expected_keywords = random.choice(self.test_data)

            # Start the tested program
            main = TestedProgram()
            main.start()

            # Send the selected planet name as input and get the output
            output = main.execute(planet_name)

            # Check if the output contains at least one of the expected keywords (case-insensitive)
            if not re.search(expected_keywords, output, re.IGNORECASE):
                return CheckResult.wrong(
                    "The output does not contain any of the expected information about the planet " + planet_name + ". Did you print the response from the model?"
                )

        return CheckResult.correct()

if __name__ == '__main__':
    LangchainTest().run_tests()
