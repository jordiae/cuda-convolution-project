CC = gcc
OBJECTS = imfilter.c
LIBS =
CFLAGS = -Wall -Wextra -O2 -std=c99
BINDIR = $(DESTDIR)/usr/bin
NAME = imfilter

imfilter: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(NAME) $(OBJECTS) $(LIBS)

clean:
	rm $(NAME)
