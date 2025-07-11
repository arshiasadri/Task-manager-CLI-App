import pygame
import random
import sys

from pygame.examples.go_over_there import clock
from pygame.examples.moveit import HEIGHT

pygame.init()

WIDTH,HEIGHT = 400,400
DICE_SIZE = 200
DOT_RADIUS = 15
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("DICE")
font = pygame.font.SysFont("Arial",24)

def draw_dice(number,x,y):
    """Draw the dice face with dots based on the number."""
    pygame.draw.rect(screen,WHITE,(x,y,DICE_SIZE,DICE_SIZE))
    pygame.draw.rect(screen,BLACK,(x,y,DICE_SIZE,DICE_SIZE),2)

    dot_positions ={
        1: [(x +DICE_SIZE // 2, y+DICE_SIZE//2)],
        2: [(x + DICE_SIZE //4,y+DICE_SIZE//4), (x +3 * DICE_SIZE//4,y+3*DICE_SIZE//4)],
        3: [(x+DICE_SIZE//4,y+DICE_SIZE//4),(x+DICE_SIZE//2,y+DICE_SIZE//2),(x+3*DICE_SIZE//4,y+3*DICE_SIZE//4)],
        4: [(x+DICE_SIZE//4,y+DICE_SIZE//4),(x+3*DICE_SIZE//4,y+DICE_SIZE//4),(x+DICE_SIZE//4,y+3*DICE_SIZE//4),(x+3*DICE_SIZE//4,y+3*DICE_SIZE//4)],
        5: [(x+DICE_SIZE//4,y+DICE_SIZE//4),(x+3*DICE_SIZE//4,y+DICE_SIZE//4),(x+DICE_SIZE//2,y+3*DICE_SIZE//2),(x+DICE_SIZE//4,y+3*DICE_SIZE//4),(x+3*DICE_SIZE//4,y+3*DICE_SIZE//4)],
        6: [(x+DICE_SIZE//4,y+DICE_SIZE//4),(x+3*DICE_SIZE//4,y+DICE_SIZE//4),(x+DICE_SIZE//4,y+DICE_SIZE//2),(x+3*DICE_SIZE//4,y+DICE_SIZE//2),(x+DICE_SIZE//4,y+3*DICE_SIZE//4),(x+3*DICE_SIZE//4,y+3*DICE_SIZE//4)]
    }
    for pos in dot_positions[number]:
        pygame.draw.circle(screen,BLACK,pos,DOT_RADIUS)

def draw_button():
    """Draw a button to roll the dice."""
    button_rect = pygame.Rect(WIDTH //2-50,HEIGHT-60,100,40)
    pygame.draw.rect(screen,RED,button_rect)
    text = font.render("beriz",True,WHITE)
    screen.blit(text,(WIDTH // 2-30,HEIGHT - 50))
    return button_rect
def main():
    current_number = 1
    button_rect = None
    rolling = False
    roll_frames = 0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect and button_rect.collidepoint(event.pos):
                    rolling = True
                    roll_frames = 20
        screen.fill((50,50,50))
        draw_dice(current_number,WIDTH//2 - DICE_SIZE//2,HEIGHT//2 - DICE_SIZE //2)
        button_rect = draw_button()
        if rolling:
            current_number = random.randint(1,6)
            roll_frames -= 1
            if roll_frames <= 0:
                rolling = False

        pygame.display.flip()
        clock.tick(10)

if __name__ == "__main__":
   main()
