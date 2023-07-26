# CV-Campus-Map
Project for Visual Interfaces for Computers Course (COMS 4735) from Columbia University's School Of Engineering and Applied Science, April 2023.

The goal of this project was to explore how well a visual image can be described in human language terms. I was tasked to write a program that takes as input a two-dimensional image of the Columbia campus and then produces as output for each “target” building a non-numeric description of this object (the “what”), a non-numeric description of its location (the “where”), and a non-numeric relationship to a “source” building (the “how”). I was also tasked with evaluating the effectiveness of these natural language outputs.

As an example, a system could print out for one of the buildings: “This medium C-shaped rightmost building is near to the building which is large, square, and centrally located”. Note that this is a description of Kent with respect to Low, but neither “Kent” nor “Low” have been mentioned. This is because a user—especially a visitor—probably doesn’t know any symbolic place names. Instead, the descriptions of the buildings come in three major parts: the “what” (“medium C-shaped” / “large”), the “where” (“rightmost” / “centrally located”) and the “how” (“near”).

My job was to write a system that uses computer vision techniques plus a primitive form of filtering and inference. First, it describes the buildings’ shapes in a minimally possible way. Then it describes their spatial locations in a minimally possible way. Then it describes their relationships with each other in the least confusing way.

Required Files:
1. The first file “Campus.pgm” is a binary image of the main campus as seen from above, where a large number within the image represents a part of a named building, and zeros represent the space between buildings.
2. The second file “Labeled.pgm” is an integer-valued image based on the first, in which each building is given an encoded integer, and all the pixels belonging to the same building are encoded with the same integer; zero still means empty space. 
3. The third file “Table.txt” translates the building’s encoded integer into a semantically meaningful string.

Project Steps:
1. Raw Data
2. “What”: describing shape
3. “Where”: describing absolute space
4. “How”: describing relative space
5. Total descriptions
