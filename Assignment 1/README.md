**1.	What are Channels and Kernels (according to EVA)?**

Channel is a collection of a particular feature. It is a collection of similar items. An example may be a channel that is a collection of the image of eyes of people.
Kernel is used for extracting features from the input. A kernel may be designed to extract edges from an image. Such a kernel moves over the entire input image collecting all the edges in the input image. The kernel moves over the input image and at each step it produces a single pixel as output a pixel which is the dot product of the kernel and the portion of the input image that the kernel is sees.


**2.	Why should we (nearly) always use 3x3 kernels?**

A 3x3 kernel is nearly always used as it can be used to multiple times to create a higher order kernel such as- applying two successive 3x3 kernels generate a 5x5 and 3 successive 3x3 kernels generate a 7x7 and so on. In this way using a 3x3 kernel also reduces the no. of parameters used. A 3x3 kernel also maintains symmetry which thus helps in not only detecting an edge (for e.g.) but also detecting its location.


**3.	How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)**

99 times we need to perform 3x3 convolution operation to reach close to 1x1 from 199x199

199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1

**4.	How are kernels initialized?**

Kernels are usually initialized randomly and then during training the kernel values are updated to best fit the input data.
Other strategies used are:

*	Set all values to 1 or 0 or another constant

*	Sample from a distribution, such as a normal or uniform distribution

**5.	What happens during the training of a DNN?**

During the training of the DNN, the model first learns to detect edges and gradients. Using these it learns to detect patterns, then parts of object, then object and finally an entire scene. Each of these happens in stages with the first stage being learning of the edges and the last stage being learning the entire scene.

