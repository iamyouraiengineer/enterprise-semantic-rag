import pytest
import config.settings

@pytest.fixture(autouse = True)
def reset_settings_singleton() -> None:
    """
    Reset the global settings cache before and after each test.
    Ensures every test starts with a fresh Settings() instance.
    """
    config.settings._settings_instance = None
    yield
    config.settings._settings_instance = None


    
