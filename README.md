# SatilliteImage

This project solely a personal project. The project is to detect the number of cars in a parking lot. As of right now I **don't** have access to the **LIVE** satillite date, but I am working with old satillite data as of right now. Using satillite images, my code will detect and count the number of cars in that parking lot. I am also building on this to detect different trees, and roads. 

Main problem I have been facing is getting **live satillite data** so if anyone knows a easy way then please email me at 
kunal.aneja101@gmail.com

## Process

1. Get Lat Long Corrdiantes of the Location you want to detect
2. Get the Lisence Key for Google Maps API
3. Using Google's API, download Satillite Image of location
4. Detect the number of cars using mulitple filters 
	5. Overlay a heat map on the image to display all cars
5. Count all the cars
6. Using grouping of white pixels, detect the lanes. 
7. Check which lots are open 



## Future outlook
- Maybe try to detect number of cars on the highway
- Try to see which places have a exterme amount of sunlight, using that give the best possible position for a solar panel.
- 
