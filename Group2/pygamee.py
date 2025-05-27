import pygame
import random
import sys
import time
from pylsl import StreamInfo, StreamOutlet



info = StreamInfo('Markers', 'Markers', channel_count=1, nominal_srate=0, channel_format='string')
outlet = StreamOutlet(info)

IMAGE_FILES = ['image/image1.jpg', 'image/image2.jpg', 'image/image3.jpg', 'image/image4.jpg', 'image/image5.jpg',
               'image/image6.jpg', 'image/image7.jpg', 'image/image8.jpg', 'image/image9.jpg', 'image/image10.jpg', 'image/image11.jpg']
CROSS_SIZE = 30   # cross dimensions
WINDOW_SIZE = (800, 600)
tot_time = 5

pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Image Viewer with Clickable Cross")
font = pygame.font.SysFont(None, 40)

def draw_cross(surface, center, size, color=(255, 0, 0), width=5):
    x, y = center
    pygame.draw.line(surface, color, (x - size//2, y - size//2), (x + size//2, y + size//2), width)
    pygame.draw.line(surface, color, (x - size//2, y + size//2), (x + size//2, y - size//2), width)

def load_images(file_list):
    return [pygame.image.load(f).convert() for f in file_list]

def random_cross_position(size):
    x = random.randint(0, WINDOW_SIZE[0])
    y = random.randint(0, WINDOW_SIZE[1])
    return x, y

images = load_images(IMAGE_FILES)

for img in images[:-1]:
    img = pygame.transform.scale(img, WINDOW_SIZE)
    cross_pos = random_cross_position(CROSS_SIZE)
    cross_rect = pygame.Rect(cross_pos[0] - CROSS_SIZE//2, cross_pos[1] - CROSS_SIZE//2, CROSS_SIZE, CROSS_SIZE)

    black = pygame.transform.scale(images[10], WINDOW_SIZE)

    clicked = False
    now = time.time()

    screen.blit(img, (0, 0))
    pygame.display.flip()
    outlet.push_sample(["IMAGE_UP"])

    while time.time() - now < 5 :
        for event in pygame.event.get() :
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    screen.blit(black, (0, 0))
    draw_cross(screen, cross_pos, CROSS_SIZE)
    pygame.display.flip()
    outlet.push_sample(["IMAGE_DOWN"])

    while not clicked :
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if cross_rect.collidepoint(event.pos):
                    clicked = True

pygame.quit()
