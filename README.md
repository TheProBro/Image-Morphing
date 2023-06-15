# Image Morphing
This is a project that demonstrates the technique of transforming one image into another smoothly by creating a sequence of intermediate images. This project provides a simple implementation of image morphing using the Delaunay triangulation and linear interpolation algorithms.

## Table of Contents
* [Working](#working)
* [Installation](#installation)
* [Usage](#usage)
* [Examples](#examples)
* [Contributing](#contributing)

## Working
The morphing works on a trained ML model of face detection on humans, and uses open-cv library to run on the server.
The javascript server file runs the python file and sends over the result to the front-end for a beautiful image morphing result

## Installation
To use this project, you need to have [Node.js](https://nodejs.org) installed on your system. Follow these steps to get started:
1. Clone the repository: 
  ~~~~ 
  git clone https://github.com/TheProBro/Image-Morphing.git
  ~~~~
2. Change into the directory:
~~~~
cd Image-Morphing
~~~~
3. Install the dependencies:
~~~~
npm install
~~~~

## Usage
1. Run the server:
~~~~
node server/server.js
~~~~
2. Run the `index.html` file in `./client/`

## Examples
![Latest](https://github.com/TheProBro/Image-Morphing/assets/35802031/d220172f-d5c7-4aea-97de-c8be57b2766a)

## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. When contributing, please provide clear descriptions of the changes made and any relevant information or context.
