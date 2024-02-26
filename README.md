Delaunay Tesselation image filter

Incremental Bowyer–Watson algorithm

![emma_512](https://user-images.githubusercontent.com/3598240/234775629-a062dcbc-c031-401d-9ff4-a9184972dd0e.png)

![output](https://user-images.githubusercontent.com/3598240/234775999-390ee40f-3ec0-40cf-8ce2-131316b3e73a.png)

![output18](https://user-images.githubusercontent.com/3598240/234777361-5119ddc1-9898-408b-8f31-9cbd36e60102.png)

![output28](https://user-images.githubusercontent.com/3598240/234777388-f0282df2-5b73-4a20-aa0d-077ce35d9390.png)

![output48](https://user-images.githubusercontent.com/3598240/234777416-aa9689f3-eb24-4e30-a214-d4c83f160f9a.png)

![outputFinal](https://user-images.githubusercontent.com/3598240/234972056-b8bf4a42-e0da-487e-b32b-4de566a59fe5.png)

![outputFinal](https://user-images.githubusercontent.com/3598240/234981307-c7b2324e-2524-419f-9ad6-4750ac79e157.png)

![outputFinal](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/246a0de9-411c-4cc4-8ae6-ca32570df4f9)

![outputFinal](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/6334740e-6d79-43e9-9e8c-17d0953cbabb)

![outputFinal](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/3f1aa811-feff-4937-ac8e-fb5825f2566d)


CUDA Based Delaunay Tesselation

Seed Creation
![outputDT1](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/f606066b-6ed4-4cc8-906a-d53ad2f00970)

Voronoi Diagram Creation using 1+JFA ( jump fill algorithm ) in a ping-pong buffer

![outputDT2](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/25593294-4e3a-47d9-ac55-3d8ea71c7c5d)

Fixing island inaccuracies within the voronoi diagram

![outputDT3](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/8b891aa1-ae25-482d-81aa-fa0fd65dccf2)

Creating CH ( graham scan convex hull from original seeds )

![outputCH](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/3cf1f714-bdf8-49f0-b0bc-d46103a23ec3)

Extracting Voronoi sites

![outputDT4](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/764cc26f-ce8e-4d5c-93df-075c96bf0573)

Excluding Voronoi sites outside of CH

![outputDT5](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/871fc316-18ef-4176-a924-9a8ca7ce8971)

Parallel triangulation based on Voronoi sites + post convex hull triangulation. Some voronoi sites integral to the final
tessellation will be contained outside of the texture. So traveling along the convex hull and finding the outside delaunay
triangles particular to the voronoi sites are necessary.


![outputDTFinal](https://github.com/AryaGosuDev/DelauneyTesselationCUDA/assets/3598240/149098da-2019-43b3-8beb-1c3d86605061)

