
### Resources used for testing
Some resources needed for testing. Only 2D images. See section UserInterfaces to create a nice cube with AIA data.


```python
# Helper function to create arrays with different chunk sizes
def createSciDBArrays(chunk_sizes, overlap):
    #create empty arrays
    !iquery -antq "create array aia_image_import <x:int64,y:int64,a0:int16,a1:int16,a2:int16,a3:int16,a4:int16,a5:int16>[csv=0:*,50000,0];"
    !iquery -antq "load(aia_image_import,'/home/simon/jupyter/res/aia_import_image.opaque',-2,'opaque');"

    !iquery -antq "create array kernel_import <x:int64,y:int64,kernel:float>[csv=0:*,50000,0];"
    !iquery -antq "load(kernel_import,'/home/simon/jupyter/res/gauss_11x11.csv',-2,'csv');"
    !iquery -antq "create array gauss_blur_11x11 <kernel:float>[y=0:10,11,0; x=0:10,11,0];"                                                                                  
    !iquery -antq "insert(redimension(kernel_import,gauss_blur_11x11),gauss_blur_11x11);"

    #create different chunk sizes
    for c_size in chunk_sizes:
        !iquery -antq "create array aia_{c_size} <aia_94:int16,aia_131:int16,aia_171:int16,aia_193:int16,aia_211:int16,aia_335:int16>[x=0:4095,{c_size},{overlap},y=0:4095,{c_size},{overlap}];"
        !iquery -antq "insert(redimension(aia_image_import,aia_{c_size}),aia_{c_size});"

```


```python
createSciDBArrays([64,128,256,512],0)
```
