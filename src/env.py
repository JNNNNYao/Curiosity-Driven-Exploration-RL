import vizdoom as vzd

def env(scenarios='dense'):
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    if scenarios == 'sparse':
        game.set_doom_scenario_path("../scenarios/my_way_home_sparse.wad")
    elif scenarios == 'verySparse':
        game.set_doom_scenario_path("../scenarios/my_way_home_verySparse.wad")
    else:
        game.set_doom_scenario_path("../scenarios/my_way_home.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")
    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.GRAY8)

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(2100)
    # Makes the window appear (turned on by default)
    game.set_window_visible(False)
    # Sets the living reward (for each move) to -1
    game.set_living_reward(-0.0001)
    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    return game