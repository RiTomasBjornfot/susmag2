**Standard deviation kernel size:**
*Defines the size of the area where the standard deviation is calculated. If the kernel size 
is set to 5, the standard deviation at pixel (i, j) is calculated from the sub-matrix at (i:i+5, j:j+5).* 

**Gaussian filter size:**
*A gaussian filter is applied to the standard deviation matrix. The parameter value x sets 
the kernel as (i-x:i+x, j-x:j+x) at pixel (i, j). Note: if set to 0, no filter is applied. For detailed 
info see:* https://juliaimages.org/ImageFiltering.jl/stable/function_reference/

**Binary threshold:**
*Sets the binary threshold for the binary matrix. Pixel values under the value will be set to 1 
and pixel values under is set to 0. The Binary matrix is used to calculate the countours. 
Each countour is surrounding a cluster in the binary image.*

**Minimum contour width/height:** *Sets the minimum allowed contour size.*

**Maximum contour width/height** *Sets the maximum allowed contour size.*
