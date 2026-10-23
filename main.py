import pygame
import sys
from menu import MainMenu
from fighting_game import FightingMatch

pygame.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hackathon Fighter Alpha")
clock = pygame.time.Clock()

# Game States
STATE_MENU = "MENU"
STATE_PLAYING = "PLAYING"
current_state = STATE_MENU

# Initialize Modular Systems
main_menu = MainMenu(SCREEN_WIDTH, SCREEN_HEIGHT)
active_match = None  # Instantiated only when the match starts

running = True
while running:
    clock.tick(60)
    
    # Global Close Window Event
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
            sys.exit()

    # State Machine Router
    if current_state == STATE_MENU:
        # Draw and update the home page layout
        menu_choice = main_menu.update(events, screen)
        if menu_choice == "START_GAME":
            # Initialize a clean match instance when the button is pressed
            active_match = FightingMatch(SCREEN_WIDTH, SCREEN_HEIGHT)
            current_state = STATE_PLAYING

    elif current_state == STATE_PLAYING:
        # Hand off control to your gameplay engine module
        match_status = active_match.update(screen)
        if match_status == "BACK_TO_MENU":
            current_state = STATE_MENU

    pygame.display.update()
