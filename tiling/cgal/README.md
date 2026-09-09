# Las processing example dir
Simples examples using las and ply fileformat

##  python with laspy 
https://laspy.readthedocs.io/en/latest/#
### Reading las / writing ply
install laspy
```console
pip3 install laspy
```

run 
```console
python3 las2ply.py
```

## Using CGAL (c++)
### Reading Las, compute nromal, compute surface
Using cgal to read a LAS and produce a surface
The CGAL doc :
https://doc.cgal.org/latest/Point_set_processing_3/index.html

Surface reconstruction doc : 
https://doc.cgal.org/latest/Advancing_front_surface_reconstruction/index.html
https://doc.cgal.org/latest/Manual/tuto_reconstruction.html

The code can run 2 algorithms :
- Advanced front surface reconstruction (https://doc.cgal.org/latest/Advancing_front_surface_reconstruction/Advancing_front_surface_reconstruction_2reconstruction_structured_8cpp-example.html)
- Poisson

#### Dataset 
The algorithm can read LAS format.

#### Build :
```console
docker compose build
```

#### Run example :
From this rep, first set local variables and create output dir (at the root of the project)


```console
export MOUNT_DIR=$(dirname "$PWD")
mkdir -p ${MOUNT_DIR}/out/
```

Actually the only executable that work is "normal_and_reconstruction"

```console
// Input : file (path of the las)
           algo (0 => poisson, 1 => structured)
// Ooutput : *.{off,ply} file in the current dir
/usr/src/app/cgal/build/normal_and_reconstruction file algo
```

computing both in same exec with poisson reconstruction on the example
```console
ALGO_TYPE=0 # 0=> Poisson, 1=> Structured
LAS_FILE=${MOUNT_DIR}/datas/urban_shift.las
docker compose run --rm cgal /bin/bash -c "/usr/src/app/cgal/build/normal_and_reconstruction ${LAS_FILE} ${ALGO_TYPE} && mv *.{off,ply} ${MOUNT_DIR}/out/ 2>/dev/null"
```

### Display 
For windows :
- https://listoffreeware.com/free-ply-viewer-software-windows/

For Linux : 
- https://www.danielgm.net/cc/
- https://www.meshlab.net/
