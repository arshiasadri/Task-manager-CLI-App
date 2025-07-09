import pygame
import sys
import random


from pygame.examples.aliens import Player
from pygame.examples.go_over_there import running

pygame.init()
pygame.mixer.init()

# sound
# pygame.mixer.music.load("background.mp3")
# pygame.mixer.music.set_volume(0.4)
# pygame.mixer.music.play(-1)

# shoot_sound = pygame.mixer.Sound("shoot.wav")
# shoot_sound.set_volume(0.6)
# sound_on = True

WIDTH, HEIGHT = 800,600
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("ON/OFF")
clock = pygame.time.Clock()

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)

# player

player_width = 50
player_height = 40
player_x = WIDTH // 2 - player_width // 2
player_y = HEIGHT - player_height - 20
player_speed = 5
player_health = 3

player_bullets = []
bullet_speed = 7

# enemy

enemy_width = 50
enemy_height = 40
enemy_x = random.randint(0,WIDTH - enemy_width)
enemy_y = 50
enemy_speed_x = 3

enemy_bullets = []
enemy_bullets_speed = 5
enemy_shoot_delay = 60
enemy_timer = 0

font = pygame.font.SysFont("Arial",24)

running = True
game_over = False

while running :
    clock.tick(60)
    screen.fill(BLACK)

    for event in pygame.event.get() :
        if event.type == pygame.QUIT :
            running = False
    if not game_over :
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_x > 0 :
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - player_width :
            player_x += player_speed
        if keys[pygame.K_SPACE]:
            player_bullets.append([player_x + player_width // 2-2,player_y])
        for bullet in player_bullets :
            bullet[1] -= bullet_speed
        player_bullets = [b for b in player_bullets if b[1] > 0]
        enemy_x += enemy_speed_x
        if enemy_x <= 0 or enemy_x >= WIDTH - enemy_width :
            enemy_speed_x *= -1
        enemy_timer += 1
        if enemy_timer >= enemy_shoot_delay :
            enemy_bullets.append([enemy_x + enemy_width //2-2, enemy_y + enemy_height])
            enemy_timer = 0
        for b in enemy_bullets[:]:
            if player_x <b[0] <player_x + player_width and player_y < b[1]<player_y + player_height :
                enemy_bullets.remove(b)
                player_health -= 1
                if player_health <= 0 :
                    game_over = True
        for b in player_bullets[:]:
            if enemy_x < b[0] < enemy_x + enemy_width and enemy_y < b[1] <enemy_y + enemy_height :
                enemy_x = random.randint(0,WIDTH - enemy_width)
                player_bullets.remove(b)
        pygame.draw.rect(screen,WHITE,(player_x,player_y,player_width,player_height))
        for bullet in player_bullets :
            pygame.draw.rect(screen,RED,(bullet[0],bullet[1],5,10))
        pygame.draw.rect(screen,GREEN,(enemy_x,enemy_y,enemy_width,enemy_height))
        for b in enemy_bullets :
            pygame.draw.rect(screen,WHITE , (b[0],b[1],5,10))
        health_text = font.render(f"health :{player_health}",True,RED)
        screen.blit(health_text,(10,10))
    else:
        over_font = pygame.font.SysFont("Arial",50)
        over_text = over_font.render("Game Over!",True,RED)
        screen.blit(over_text,(WIDTH // 2 - 120, HEIGHT // 2 - 30))
    pygame.display.flip()
pygame.quit()
sys.exit()



