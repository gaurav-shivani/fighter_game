import pygame
import sys

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 1.5

# Setup Display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Python Fighter - Animated Edition")
clock = pygame.time.Clock()

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# --- NEW: LOAD BACKGROUND ASSET ---
try:
    # Load and stretch your background image to fill the canvas
    bg_image = pygame.image.load("background.png").convert_alpha()
    bg_image = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
except:
    # Fallback if image doesn't exist yet
    bg_image = None

class Fighter:
    def __init__(self, x, y, controls, flip=False):
        self.rect = pygame.Rect((x, y, 80, 160)) # Collision Hitbox
        self.vel_y = 0
        self.jump = False
        self.health = 100
        self.controls = controls
        self.flip = flip # True if facing left (Player 2 starts flipped)
        
        # Animation States
        self.action = "idle" # 'idle' or 'attack'
        self.frame_index = 0
        self.update_time = pygame.time.get_ticks()
        
        # --- NEW: LOAD ANIMATION SPRITES ---
        self.animations = {'idle': [], 'attack': []}
        self.load_sprites()
        
    def load_sprites(self):
        # Placeholder: Creating dummy color frames if assets aren't downloaded yet
        # Replace these loops later by loading real image frames from your files!
        try:
            # Example for real asset parsing:
            # self.animations['idle'].append(pygame.image.load("idle_frame1.png"))
            pass
        except:
            pass
            
        # Creating colored surface fallbacks so the game runs immediately
        for i in range(4): # 4 frames of idle
            surf = pygame.Surface((120, 180), pygame.SRCALPHA)
            pygame.draw.rect(surf, (0, 100, 255) if not self.flip else (255, 50, 50), (20, 20, 80, 160))
            # Draw a tiny indicator "head" to show directional facing
            pygame.draw.circle(surf, (255, 255, 255), (90 if not self.flip else 30, 50), 15)
            self.animations['idle'].append(surf)
            
        for i in range(3): # 3 frames of attacking
            surf = pygame.Surface((170, 180), pygame.SRCALPHA)
            pygame.draw.rect(surf, (0, 200, 255) if not self.flip else (255, 150, 150), (20 if self.flip else 70, 20, 80, 160))
            # Draw an extended fist box representing the punch visually
            pygame.draw.rect(surf, GREEN, (100 if not self.flip else 0, 70, 50, 30))
            self.animations['attack'].append(surf)

    def move(self, target):
        dx = 0
        keys = pygame.key.get_pressed()

        # Face toward the enemy target dynamically
        if self.rect.centerx < target.rect.centerx:
            self.flip = False
        else:
            self.flip = True

        if self.action != "attack":
            if keys[self.controls['left']]:
                dx = -8
            if keys[self.controls['right']]:
                dx = 8
            if keys[self.controls['jump']] and not self.jump:
                self.vel_y = -30
                self.jump = True

        # Apply gravity
        self.vel_y += GRAVITY
        dy = self.vel_y

        # Floor boundaries
        if self.rect.bottom + dy > SCREEN_HEIGHT - 60:
            dy = (SCREEN_HEIGHT - 60) - self.rect.bottom
            self.jump = False
            self.vel_y = 0

        self.rect.x += dx
        self.rect.y += dy

    def update_animation(self):
        # Dynamic timer cycle to progress structural frames
        ANIMATION_COOLDOWN = 120 # Milliseconds per frame
        
        # Grab current rendering frame loop
        self.image = self.animations[self.action][self.frame_index]
        
        if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
            
        # Reset animation sequence at end of cycle
        if self.frame_index >= len(self.animations[self.action]):
            if self.action == "attack":
                self.action = "idle" # Return to neutral state
            self.frame_index = 0

    def attack(self, target):
        keys = pygame.key.get_pressed()
        if keys[self.controls['attack']] and self.action != "attack":
            self.action = "attack"
            self.frame_index = 0
            
            # Generate target check region
            attack_rect = pygame.Rect(self.rect.right if not self.flip else self.rect.left - 60, self.rect.y + 40, 60, 40)
            if attack_rect.colliderect(target.rect):
                target.health -= 7

    def draw(self, surface):
        # Center image visualization offset onto the rigid physics bounding box
        img_rect = self.image.get_rect()
        img_rect.center = (self.rect.centerx, self.rect.centery - 10)
        surface.blit(self.image, img_rect)

def draw_hud(p1, p2):
    pygame.draw.rect(screen, RED, (50, 30, 400, 25))
    pygame.draw.rect(screen, RED, (550, 30, 400, 25))
    pygame.draw.rect(screen, GREEN, (50, 30, max(0, p1.health * 4), 25))
    pygame.draw.rect(screen, GREEN, (550, 30, max(0, p2.health * 4), 25))

# Controls mapping updated for standard Mac laptop keyboards
p1_inputs = {'left': pygame.K_a, 'right': pygame.K_d, 'jump': pygame.K_w, 'attack': pygame.K_f}
p2_inputs = {'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'jump': pygame.K_UP, 'attack': pygame.K_l}

player1 = Fighter(200, 380, p1_inputs, flip=False)
player2 = Fighter(700, 380, p2_inputs, flip=True)

# Main Game Loop
running = True
while running:
    clock.tick(FPS)
    
    # Render Stage Background
    if bg_image:
        screen.blit(bg_image, (0, 0))
    else:
        screen.fill((50, 50, 50)) # Fallback safe color
        pygame.draw.rect(screen, (100, 100, 100), (0, SCREEN_HEIGHT - 60, SCREEN_WIDTH, 60))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()

    # Engine Logic Loops
    player1.move(player2)
    player2.move(player1)
    
    player1.attack(player2)
    player2.attack(player1)
    
    player1.update_animation()
    player2.update_animation()

    # Render Frame Entities
    player1.draw(screen)
    player2.draw(screen)
    draw_hud(player1, player2)

    pygame.display.update()
