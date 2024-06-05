# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selenium based web UI tests."""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait


# A dictionary of constants to be used in the test
test_constants = {
    "PROJECT_NAME": "TLDRLC",
    "PROJECT_DEPLOY_URL": "http://localhost:8765",
    "STEP1_BTN_TEXT_LEARN_MORE": "learn more",
    "STEP1_BTN_TEXT_CONTINUE": "continue",
    "STEP2_BTN_TEXT_GET_DATA": "get data",
    "STEP3_BTN_TEXT_LETS_CHAT": "let's chat",
}


def test_initial_ui_loading():
    """Test if the initial UI loads correctly."""
    driver = webdriver.Firefox()
    driver.get(test_constants["PROJECT_DEPLOY_URL"])
    driver.maximize_window()

    # Implicit wait will not work as the title will change dynamically
    # driver.implicitly_wait(15)
    wait = WebDriverWait(driver, 10)
    wait.until(lambda driver: test_constants["PROJECT_NAME"] in driver.title)

    # Confirmations
    # Page title is displayed correctly when loading finishes.
    assert test_constants["PROJECT_NAME"] in driver.title

    # Check if the "Continue" button is available.
    buttons_step = driver.find_elements(By.TAG_NAME, "button")
    for button in buttons_step:
        button_text = button.text.lower()
        if button_text == test_constants["STEP1_BTN_TEXT_CONTINUE"]:
            assert button.is_displayed()
            # Click the button
            button.click()

    # Check if the "Get data" button is available.
    buttons_step = driver.find_elements(By.TAG_NAME, "button")
    for button in buttons_step:
        button_text = button.text.lower()
        if button_text == test_constants["STEP2_BTN_TEXT_GET_DATA"]:
            assert button.is_displayed()
            # Click the button
            button.click()

    # Check if the "Let's chat" button is available.
    buttons_step = driver.find_elements(By.TAG_NAME, "button")
    for button in buttons_step:
        button_text = button.text.lower()
        if button_text == test_constants["STEP3_BTN_TEXT_LETS_CHAT"]:
            assert button.is_displayed()
            # We should not be able go further as the chatbot is not initialised yet.
            assert not button.is_enabled()

    driver.quit()


# We do not need it if these tests are called using pytest
if __name__ == "__main__":
    test_initial_ui_loading()
