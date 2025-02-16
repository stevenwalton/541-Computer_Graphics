#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>

#include <algorithm>
#include <cmath>

using std::cerr;
using std::endl;
// Screen size globals
static int Nx = 1786;
static int Ny = 1344;

double ceil_441(double f)
{
    return ceil(f-0.00001);
}

double floor_441(double f)
{
    return floor(f+0.00001);
}


vtkImageData *
NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    return img;
}

void
WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}

class Screen 
{
  public:
      // Constructor
      Screen(int w, int h) {width = w; height = h;}
      // Values
      unsigned char   *buffer;
      int             width, height;
      // Functions
      void SetPixel(int c, int r, unsigned char *color);
};

void
Screen::SetPixel(int c, int r, unsigned char *color)
{
    if (r < 0 || r >= height || c < 0 || c>= width)
        return;
    int index = ((r*width) + c)*3;
    buffer[index+0] = color[0];
    buffer[index+1] = color[1];
    buffer[index+2] = color[2];
}

// Make a global object
Screen screen(Nx, Ny);

class Triangle
{
    public:
        // Values
        double          X[3];
        double          Y[3];
        unsigned char   color[3];
        double          minX, maxX, minY, maxY;
        int             minY_index, maxY_index, middleY_index;
        // Functions
        void            raster();
        void            getMinMax();
        double          getXIntercept();
        void            splitTriangles(); 
        void            drawTriangle(double lX[3], double lY[3]); 
};

void
Triangle::raster()
{
    getMinMax();
    splitTriangles();
}

void
Triangle::getMinMax()
{
    minX = *std::min_element(X,X+3);
    maxX = *std::max_element(X,X+3);
    
    minY = *std::min_element(Y,Y+3);
    maxY = *std::max_element(Y,Y+3);
    
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == minY) minY_index = i;
        else if(Y[i] == maxY) maxY_index = i;
        else middleY_index = i;
    }
}

double
Triangle::getXIntercept()
{
    double m = (Y[maxY_index] - Y[minY_index]) / (X[maxY_index] - X[minY_index]);
    double b = Y[maxY_index] - (m * X[maxY_index]);
    return ( (Y[middleY_index] - b) / m);
}

void
Triangle::splitTriangles()
{
    // Set X in correct order for easier math
    double xIntercept = getXIntercept();
    // Lower Triangle
    double lX[3] = {X[middleY_index], xIntercept, X[minY_index]};
    double lY[3] = {Y[middleY_index], Y[middleY_index], minY};
    drawTriangle(lX, lY);
    // Upper triangle
    double uX[3] = {X[middleY_index], xIntercept, X[maxY_index]};
    double uY[3] = {Y[middleY_index], Y[middleY_index], maxY};
    drawTriangle(uX, uY);
}

void
Triangle::drawTriangle(double lX[3], double lY[3])
{
    double lminX = *std::min_element(lX, lX+3);
    double lmaxX = *std::max_element(lX, lX+3);

    double lminY = *std::min_element(lY, lY+3);
    double lmaxY = *std::max_element(lY, lY+3);
    // Adjust triangle
    if(lX[0] > lX[1])
    {
        double x0,x1;
        x0 = lX[1];
        x1 = lX[0];
        lX[0] = x0;
        lX[1] = x1;
    }
    double m0,m1,b0,b1,y0,y1;
    if(lX[0] != lX[2]) // right triangle
    {
        m0 = (lY[2] - lY[0]) / (lX[2] - lX[0]);
        b0 = lY[0] - (m0*lX[0]);
    }
    if(lX[1] != lX[2]) // right triangle
    {
        m1 = (lY[2] - lY[1]) / (lX[2] - lX[1]);
        b1 = lY[1] - (m1*lX[1]);
    }
    for(int row = ceil_441(lminY); row <= floor_441(lmaxY); ++row)
    {
        double leftX, rightX;
        leftX = (row - b0)/m0;
        rightX = (row - b1)/m1;
        // Only loop over x pixels within triangle
        for(int col = ceil_441(leftX); col <= floor_441(rightX); ++col)
        {
            screen.SetPixel(col,row,color);
        }
    }
}

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1c_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();
    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkFloatArray *colors = (vtkFloatArray *) pd->GetPointData()->GetArray("color_nodal");
    float *color_ptr = colors->GetPointer(0);
    std::vector<Triangle> tris(numTris);
    vtkIdType npts;
    vtkIdType *ptIds;
    int idx;
    for (idx = 0, cells->InitTraversal() ; cells->GetNextCell(npts, ptIds) ; idx++)
    {
        if (npts != 3)
        {
            cerr << "Non-triangles!! ???" << endl;
            exit(EXIT_FAILURE);
        }
        tris[idx].X[0] = pts->GetPoint(ptIds[0])[0];
        tris[idx].X[1] = pts->GetPoint(ptIds[1])[0];
        tris[idx].X[2] = pts->GetPoint(ptIds[2])[0];
        tris[idx].Y[0] = pts->GetPoint(ptIds[0])[1];
        tris[idx].Y[1] = pts->GetPoint(ptIds[1])[1];
        tris[idx].Y[2] = pts->GetPoint(ptIds[2])[1];
        tris[idx].color[0] = (unsigned char) color_ptr[4*ptIds[0]+0];
        tris[idx].color[1] = (unsigned char) color_ptr[4*ptIds[0]+1];
        tris[idx].color[2] = (unsigned char) color_ptr[4*ptIds[0]+2];
    }
    cerr << "Done reading" << endl;

    return tris;
}

int main()
{
   vtkImageData *image = NewImage(Nx, Ny);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   int npixels = Nx*Ny;
   // Initialize everything as black
   for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;

   std::vector<Triangle> triangles = GetTriangles();
   
   // Screen is a global object
   screen.buffer = buffer;

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM
   for (int i = 0; i < triangles.size(); ++i)
       triangles[i].raster();
   WriteImage(image, "allTriangles");
}
