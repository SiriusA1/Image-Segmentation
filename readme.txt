This program takes 2 png images, "white-tower.png" and "wt_slic.png".
They must have those names or the program will not work. When running the program
is does not require any inputs, as long as the images are in the same directory it will
work properly.

The program will output 2 png images, "k-means_tower.png" and "SLIC_tower_borders.png", 
however I have included a third image called "SLIC_tower.png" so it's easier to see
the different clusters without all the noise from the black borders (moreso for the
tower and tree areas).

The k-means implementation uses a list of 10 dictionaries (one for each center)
with the keys being the distance from the centroid and the values for the keys being
the points with that distance from the centroid. This implementation is a bit useless
now as I originally needed the sorted dictionary to reassign centroids to the median 
point of the cluster, but the program now only uses the mean. K-means iterates through each pixel
in the image and assigns them to the closest centroid with respect to RGB values, then it updates
the centroids by taking the mean and checks to see if the new centroid is within a threshold of
the old centroid. If it is, the program stops and if it isn't the process iterates again.

The SLIC implementation uses multiple dictionaries, one that has the centroids as keys and all the points,
including their rgb values, in the cluster as values, and another that has every point as a key with its
corresponding centroid and distance from the centroid as the single value. The method starts by evenly
initializing centroids for 50x50 squares and then it takes the gradient around the centroids and locally
shifts them. Next the proper loop begins and the program begins creating the clusters. Instead of iterating
through each pixel, the loop iterates around the neighborhood of each centroid and assigns points. If a point
has already been assigned to a centroid and a closer one is found, the dictionaries are updated to remove
the point from the old centroid dictionary and to update the point's new centroid. After clusters are
formed, new centroids are created from the mean x, y, and RGB of all the points in the cluster. The
program checks to see if the new centroids are within a certain threshold of the old centroid just like
the k-means method, but this method had a maximum of 3 iterations. Then the method colors the pixels 
of each cluster with the mean RGB value and colors pixels near neighboring clusters with black.