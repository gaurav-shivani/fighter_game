import pygame

class MainMenu:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("Arial", 36, bold=True)
        self.start_btn = pygame.Rect((width // 2) - 150, (height // 2) - 30, 300, 60)

    def update(self, events, surface):
        surface.fill((20, 24, 33))
        mouse_pos = pygame.mouse.get_pos()

        # Title
        title_surf = self.font.render("MOTION KOMBAT", True, (255, 255, 255))
        surface.blit(title_surf, ((self.width // 2) - (title_surf.get_width() // 2), 140))

        # Button Hover & Drawing
        is_hovered = self.start_btn.collidepoint(mouse_pos)
        btn_color = (245, 75, 85) if is_hovered else (230, 57, 70)
        pygame.draw.rect(surface, btn_color, self.start_btn, border_radius=8)

        btn_text = self.font.render("PLAY", True, (255, 255, 255))
        text_rect = btn_text.get_rect(center=self.start_btn.center)
        surface.blit(btn_text, text_rect)

        # Triggers
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            return "START_GAME"

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_hovered:
                    return "START_GAME"

        return None