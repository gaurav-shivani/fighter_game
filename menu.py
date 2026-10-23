import pygame

class MainMenu:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("Arial", 40)
        # Quick clickable button bounding box centered horizontally
        self.start_btn = pygame.Rect((width // 2) - 150, (height // 2) - 30, 300, 60)

    def update(self, events, surface):
        surface.fill((20, 24, 33)) # Dark Hackathon Aesthetic Background
        
        # Draw a clean placeholder title
        title_surf = self.font.render("MORTAL KOMBAT CLONE", True, (255, 255, 255))
        surface.blit(title_surf, ((self.width // 2) - (title_surf.get_width() // 2), 100))
        
        # Render a simple interactive button box
        pygame.draw.rect(surface, (230, 57, 70), self.start_btn, border_radius=8)
        btn_text = self.font.render("PRESS ENTER TO PLAY", True, (255, 255, 255))
        surface.blit(btn_text, ((self.width // 2) - (btn_text.get_width() // 2), (self.height // 2) - 20))

        # Check for activation inputs
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            return "START_GAME"
            
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.start_btn.collidepoint(event.pos):
                    return "START_GAME"
        return None
