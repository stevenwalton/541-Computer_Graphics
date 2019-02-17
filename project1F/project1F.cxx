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
#include <vtkDoubleArray.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#define NORMALS

using std::cerr;
using std::endl;
// Screen size globals
static int Nx = 1000;
static int Ny = 1000;

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
   //full_filename = "Images/" + full_filename;
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}

struct LightingParameters
{
    LightingParameters(void)
    {
         lightDir[0] = -0.6;
         lightDir[1] = 0;
         lightDir[2] = -0.8;
         Ka = 0.3;
         Kd = 0.7;
         Ks = 2.3;
         alpha = 2.5;
    };
  

    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
};

LightingParameters lp;

class Matrix
{
  public:
    double          A[4][4];

    void            TransformPoint(const double *ptIn, double *ptOut);
    static Matrix   ComposeMatrices(const Matrix &, const Matrix &);
    void            Print(ostream &o);
};

void
Matrix::Print(ostream &o)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        char str[256];
        sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
        o << str;
    }
}

Matrix
Matrix::ComposeMatrices(const Matrix &M1, const Matrix &M2)
{
    Matrix rv;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            rv.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                rv.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }

    return rv;
}


void
Matrix::TransformPoint(const double *ptIn, double *ptOut)
{
    ptOut[0] = ptIn[0]*A[0][0]
             + ptIn[1]*A[1][0]
             + ptIn[2]*A[2][0]
             + ptIn[3]*A[3][0];
    ptOut[1] = ptIn[0]*A[0][1]
             + ptIn[1]*A[1][1]
             + ptIn[2]*A[2][1]
             + ptIn[3]*A[3][1];
    ptOut[2] = ptIn[0]*A[0][2]
             + ptIn[1]*A[1][2]
             + ptIn[2]*A[2][2]
             + ptIn[3]*A[3][2];
    ptOut[3] = ptIn[0]*A[0][3]
             + ptIn[1]*A[1][3]
             + ptIn[2]*A[2][3]
             + ptIn[3]*A[3][3];
}


class Screen 
{
  public:
      // Constructor
      Screen(int w, int h) {width = w; height = h;}
      // Values
      unsigned char   *buffer;
      double          *zbuffer;
      int             width, height;
      // Functions
      void SetPixel(int c, int r, double z, double color[3], double s);
};


void
Screen::SetPixel(int c, int r, double z, double color[3], double s)
{
    int pixel = ((r*width) + c);
    if (r < 0 || r >= height || c < 0 || c>= width)
        return;
    /*
    if(z < -1 || z > 0)
    {
        cerr << "ERROR: z=" << z << " which is out of bounds" << endl;
        abort();
    }
    */
    if(z < zbuffer[pixel])
        return;
    zbuffer[pixel] = z;
    int index = pixel*3;
    for(int i = 0; i < 3; ++i)
    {
        double c = color[i]*s;
        if (c > 1) c = 1;
        buffer[index+i] = ceil_441(c*255.);
    }
}

// Make a global object
Screen screen(Nx, Ny);

class Camera
{
  public:
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];

    Matrix          ViewTransform(void);
    Matrix          CameraTransform(void);
    Matrix          DeviceTransform(void);
    Matrix          VT_CT_DT(void);
};


double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

Camera
GetCamera(int frame, int nframes)
{
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0;
    c.focus[1] = 0;
    c.focus[2] = 0;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}

Matrix
Camera::ViewTransform()
{
    double cot = 1./(tan(angle/2.));
    double pos22 = (far+near)/(far-near);
    double pos32 = (2*far*near)/(far-near); 
    Matrix m;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            m.A[i][j] = 0.;
    m.A[0][0] = cot;
    m.A[1][1] = cot;
    m.A[2][2] = pos22;
    m.A[3][2] = pos32;
    m.A[2][3] = -1;
    return m;
}

Matrix
Camera::CameraTransform()
{
    Matrix m;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            m.A[i][j] = 0.;

    // Normalize all dem vectors!
    // w
    double w[3];
    for(int i = 0; i < 3; ++i)
        w[i] = position[i] - focus[i];
    double wMag = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

    for(int i = 0; i < 3; ++i)
        w[i] = w[i]/wMag;

    // u
    double u[3] = {(up[1]*w[2] - up[2]*w[1]),
                   (up[2]*w[0] - up[0]*w[2]),
                   (up[0]*w[1] - up[1]*w[0])};
    double uMag = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    for(int i = 0; i < 3; ++i)
        u[i] = u[i]/uMag;

    // v (not up!)
    double v[3] = {w[1]*u[2] - w[2]*u[1],
                   w[2]*u[0] - w[0]*u[2],
                   w[0]*u[1] - w[1]*u[0]};
    double vMag = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    for(int i = 0; i < 3; ++i)
        v[i] = v[i]/vMag;


    m.A[0][0] = u[0];
    m.A[0][1] = v[0];
    m.A[0][2] = w[0];

    m.A[1][0] = u[1];
    m.A[1][1] = v[1];
    m.A[1][2] = w[1];

    m.A[2][0] = u[2];
    m.A[2][1] = v[2];
    m.A[2][2] = w[2];

    m.A[3][0] = -1*(u[0]*position[0] + u[1]*position[1] + u[2]*position[2]);
    m.A[3][1] = -1*(v[0]*position[0] + v[1]*position[1] + v[2]*position[2]);
    m.A[3][2] = -1*(w[0]*position[0] + w[1]*position[1] + w[2]*position[2]);
    m.A[3][3] = 1;

    return m;
}

Matrix
Camera::DeviceTransform()
{
    Matrix dt;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            dt.A[i][j] = 0;

    // coords * M
    dt.A[0][0] = screen.width/2;
    dt.A[3][0] = screen.width/2;
    dt.A[1][1] = screen.height/2;
    dt.A[3][1] = screen.height/2;
    dt.A[2][2] = 1; // Z isn't transformed
    dt.A[3][3] = 1;
    return dt;
}

Matrix
Camera::VT_CT_DT()
{
    Matrix vt = ViewTransform();
    Matrix ct = CameraTransform();
    Matrix dt = DeviceTransform();
    Matrix vtct = vt.ComposeMatrices(ct,vt);
    Matrix m = vtct.ComposeMatrices(vtct,dt);
    return m;
}

class Triangle
{
    public:
        // Values
        double          X[3];
        double          Y[3];
        double          Z[3];
        double          colors[3][3];
        double          normals[3][3];
        double          shading[3];
        double          ymin, ymax;
        int             ymin_index, ymax_index, ymid_index;
        Camera          c; // Functions
        inline void     setCamera(Camera cam) { c = cam;};
        void            raster();
        void            getMinMax();
        bool            isArbitrary();
        double          getIntercept(double x[3], double y[3]);
        void            splitTriangle();
        void            drawTriangle(double lX[3], double lY[3], double lZ[3], double c[3][3], double shad[3]);
        void            orientTriangleAndDraw();
        void            lerp(double x0, double x1, double x2, 
                             double f0[3], double f1[3],
                             double color[3]);
        double          lerp(double x0, double x1, double x2, 
                             double f0, double f1);
        Matrix          triangle2Matrix();
        void            Print(ostream &o);
        void            matrix2Triangle(Matrix m);
        void            applyShading();
        void            ambient(void);
        double          diffuse(LightingParameters &, double *normals);
        void            specular(void);
        double          calculatePhongShading(LightingParameters &,
                                              double *vert,
                                              Camera &c,
                                              double *n);
};

void
Triangle::raster()
{
    getMinMax();
    if(isArbitrary())
        splitTriangle();
    else
        orientTriangleAndDraw();
}

void
Triangle::getMinMax()
{
    ymin = *std::min_element(Y,Y+3);
    ymax = *std::max_element(Y,Y+3);
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] != ymin && Y[i] != ymax)
            ymid_index = i;
        else if (Y[i] == ymin)
            ymin_index = i;
        else if (Y[i] == ymax)
            ymax_index = i;
    }
}

void
Triangle::applyShading()
{
    for(int i = 0; i < 3; ++i)
    {
        double vert[3] = {X[i],Y[i],Z[i]};
        //double phong = calculatePhongShading(lp, vert,c, normals[i]);
        shading[i] = calculatePhongShading(lp, vert, c, normals[i]);

        //for(int j = 0; j < 3; ++j)
        //{
        //    colors[i][j] *= phong;
        //    if(colors[i][j] > 1)
        //        colors[i][j] = 1;
        //}
    }
}

bool
Triangle::isArbitrary()
{
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == Y[(i+1)%3])
            return false;
    }
    return true;
}

void
Triangle::orientTriangleAndDraw()
{
    // Check if going down
    double x[3];
    double y[3];
    double z[3];
    double s[3];
    double c[3][3];
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == Y[(i+1)%3])
        {
            x[0] = X[i];
            y[0] = Y[i];
            z[0] = Z[i];
            s[0] = shading[i];
            c[0][0] = colors[i][0];
            c[0][1] = colors[i][1];
            c[0][2] = colors[i][2];

            x[1] = X[(i+1)%3];
            y[1] = Y[(i+1)%3];
            z[1] = Z[(i+1)%3];
            s[1] = shading[(i+1)%3];
            c[1][0] = colors[(i+1)%3][0];
            c[1][1] = colors[(i+1)%3][1];
            c[1][2] = colors[(i+1)%3][2];

            x[2] = X[(i+2)%3];
            y[2] = Y[(i+2)%3];
            z[2] = Z[(i+2)%3];
            s[2] = shading[(i+2)%3];
            c[2][0] = colors[(i+2)%3][0];
            c[2][1] = colors[(i+2)%3][1];
            c[2][2] = colors[(i+2)%3][2];
        }
    }
    if(!(y[2] == ymin || y[2] == ymax))
    {
        cerr << "y's aren't right" << endl;
        abort();
    }
    drawTriangle(x,y,z,c,s);
}

double
Triangle::getIntercept(double x[3], double y[3])
{
    double lmin = *std::min_element(y,y+3);
    double lmax = *std::max_element(y,y+3);
    int minIndex,midIndex, maxIndex;
    // Assumes there is a midpoint
    for(int i = 0; i < 3; ++i)
    {
        if (y[i] != lmin && y[i] != lmax)
            midIndex = i;
        else if(y[i] == lmin)
            minIndex = i;
        else if(y[i] == lmax)
            maxIndex = i;
    }
    if(x[maxIndex] == x[minIndex])
        return x[maxIndex];
    double m = (y[maxIndex] - y[minIndex])/(x[maxIndex] - x[minIndex]);
    double b = y[maxIndex] - (m*x[maxIndex]);
    return ((y[midIndex] - b)/m);
}

void
Triangle::splitTriangle()
{
    double xintercept = getIntercept(X,Y);
    double lX[3] = {X[ymid_index], xintercept, X[ymin_index]};
    double lY[3] = {Y[ymid_index], Y[ymid_index], Y[ymin_index]};

    // Z intercept
    double zintercept = lerp(Y[ymax_index], Y[ymin_index], Y[ymid_index], Z[ymax_index], Z[ymin_index]);
    double lZ[3] = {Z[ymid_index], zintercept, Z[ymin_index]};

    // Colors intercept
    double cintercept[3] = {0,0,0};
    lerp(Y[ymax_index], Y[ymin_index], Y[ymid_index], colors[ymax_index], colors[ymin_index], cintercept);
    double lC[3][3];
    lC[0][0] = colors[ymid_index][0];
    lC[0][1] = colors[ymid_index][1];
    lC[0][2] = colors[ymid_index][2];
    lC[1][0] = cintercept[0];
    lC[1][1] = cintercept[1];
    lC[1][2] = cintercept[2];
    lC[2][0] = colors[ymin_index][0];
    lC[2][1] = colors[ymin_index][1];
    lC[2][2] = colors[ymin_index][2];

    // get shading at intercept
    double shadingIntercept = lerp(Y[ymax_index], Y[ymin_index], Y[ymid_index], shading[ymax_index], shading[ymin_index]);
    double lShading[3] = {shading[ymid_index], shadingIntercept, shading[ymin_index]};

    // Double check that is lower
    if(lY[2] != ymin)
    {
        cerr << "Not a lower triangle" << endl;
        cerr << "ymin: " << ymin << endl;
        cerr << "lY[2]: " << lY[2] << endl;
        cerr << (lY[2] == ymin) << endl;
        cerr << (lY[2] != ymin) << endl;
        cerr << "Original:" << endl;
        cerr << X[0] << " " << Y[0] << "\n"
             << X[1] << " " << Y[1] << "\n"
             << X[2] << " " << Y[2] << endl;
        cerr << endl;
        cerr << "'lower' triangle" << endl;
        cerr << lX[0] << " " << lY[0] << "\n"
             << lX[1] << " " << lY[1] << "\n"
             << lX[2] << " " << lY[2] << endl;
        abort();
    }
    drawTriangle(lX,lY,lZ,lC,lShading);
    double uX[3] = {X[ymid_index], xintercept, X[ymax_index]};
    double uY[3] = {Y[ymid_index], Y[ymid_index], Y[ymax_index]};
    double uZ[3] = {Z[ymid_index], zintercept, Z[ymax_index]};
    double uShading[3] = {shading[ymid_index], shadingIntercept, shading[ymax_index]};

    double uC[3][3];
    uC[0][0] = colors[ymid_index][0];
    uC[0][1] = colors[ymid_index][1];
    uC[0][2] = colors[ymid_index][2];
    uC[1][0] = cintercept[0];
    uC[1][1] = cintercept[1];
    uC[1][2] = cintercept[2];
    uC[2][0] = colors[ymax_index][0];
    uC[2][1] = colors[ymax_index][1];
    uC[2][2] = colors[ymax_index][2];
    // Double check that is upper
    if(uY[2] != ymax)
    {
        cerr << "Not a upper triangle" << endl;
        abort();
    }
    drawTriangle(uX,uY,uZ,uC,uShading);
}
  
// For colors
void
Triangle::lerp(double x0, double x1, double x2,             // coordinates
               double f0[3], double f1[3],    // field values
               double f2[3])
{
    double t = 0;
    if (x1 != x0) t = (x2-x0)/(x1-x0);
    f2[0] = f0[0] + t*(f1[0]-f0[0]);
    f2[1] = f0[1] + t*(f1[1]-f0[1]);
    f2[2] = f0[2] + t*(f1[2]-f0[2]);
}

// For Z values
double
Triangle::lerp(double x0, double x1, double x2,             // coordinates
               double f0, double f1)    // field values
{
    double t = 0;
    if (x1 != x0) t = (x2-x0)/(x1-x0);
    double f = (f0 + t*(f1-f0));
    return f;
}

void
Triangle::drawTriangle(double x[3], double y[3], double z[3], double c[3][3], double s[3])
{
    int lowerY = ceil_441(*std::min_element(y,y+3));
    int upperY = floor_441(*std::max_element(y,y+3));
    // Check that orientation is correct
    if(y[0] != y[1])
    {
        cerr << "Triangle isn't lower or upper" << endl;
        cerr 
             << x[0] << " " << y[0] << "\n"
             << x[1] << " " << y[1] << "\n"
             << x[2] << " " << y[2] 
             << endl;
        abort();
    }
    // Check that upper or lower
    if((y[2] - ymax)>1e6 || (y[2] - ymin)>1e6)
    {
        cerr << "y2 is not a max or min" << endl;
        cerr << "y2: " << y[2] << endl;
        cerr << ymin << " " << ymax << endl;
        abort();
    }
    // Set x[0] on left side
    if(x[1] < x[0])
    {
        double x0 = x[0];
        double z0 = z[0];
        double c0[3] = {c[0][0], c[0][1], c[0][2]};
        double s0 = s[0];
        x[0] = x[1];
        x[1] = x0;
        z[0] = z[1];
        z[1] = z0;
        s[0] = s[1];
        s[1] = s0;
        c[0][0] = c[1][0];
        c[0][1] = c[1][1];
        c[0][2] = c[1][2];
        c[1][0] = c0[0];
        c[1][1] = c0[1];
        c[1][2] = c0[2];
    }
    double m0 = 0;
    double m1 = 0;
    double b0, b1;
    // Get slopes for left and right if available
    if(x[0] != x[2]) // Not a right triangle with left vertical 
    {
        m0 = (y[2]-y[0])/(x[2]-x[0]);
        b0 = y[2] - (m0*x[2]);
    }
    if(x[1] != x[2]) // Not a right triangle with right vertical
    {
        m1 = (y[2]-y[1])/(x[2]-x[1]);
        b1 = y[2] - (m1*x[2]);
    }

    // Row loop
    for(int row = lowerY; row <= upperY; ++row)
    {
        double leftX, rightX;
        // Get leftX and rightX considering if right triangles or not
        if(m0 == 0)
            leftX = *std::min_element(x,x+3);
        else
            leftX = (row - b0)/m0;
        if(m1 == 0)
            rightX = *std::max_element(x,x+3);
        else
            rightX = (row - b1)/m1;
        // LERPing in the y's
        double lz, rz, ls, rs;
        double fl[3], fr[3];
        lz = lerp(y[2], y[0], row, z[2], z[0]);
        rz = lerp(y[2], y[1], row, z[2], z[1]);
        lerp(y[2], y[0], row, c[2], c[0], fl); // left color
        lerp(y[2], y[1], row, c[2], c[1], fr); // right color
        ls = lerp(y[2], y[0], row, s[2], s[0]);
        rs = lerp(y[2], y[1], row, s[2], s[1]);
        // Col loop
        for(int col = ceil_441(leftX); col <= floor_441(rightX); ++col)
        {
            double color[3] = {0,0,0};
            double z = lerp(leftX, rightX, col, lz, rz);
            double shad = lerp(leftX, rightX, col, ls, rs);
            lerp(leftX, rightX, col, fl, fr, color);    // color
            screen.SetPixel(col,row,z,color,shad);
        }
    }
}

Matrix
Triangle::triangle2Matrix()
{
    Matrix m;
    for(int i = 0; i < 3; ++i)
    {
        m.A[i][0] = X[i];
        m.A[i][1] = Y[i];
        m.A[i][2] = Z[i];
        m.A[i][3] = 1;
    }
    for(int i = 0; i < 4; ++i)
        m.A[3][i] = 0;
    return m;
}

void
Triangle::matrix2Triangle(Matrix m)
{
    for(int i = 0; i < 3; ++i)
    {
        X[i] = m.A[0][i];
        Y[i] = m.A[1][i];
        Z[i] = m.A[2][i];
    }
}

void
Triangle::Print(ostream &o)
{
    char strX[256];
    char strY[256];
    char strZ[256];
    char strC0[256];
    char strC1[256];
    char strC2[256];
    sprintf(strX, "X: (%.7f, %.7f, %.7f)\n", X[0], X[1], X[2]);
    sprintf(strY, "Y: (%.7f, %.7f, %.7f)\n", Y[0], Y[1], Y[2]);
    sprintf(strZ, "Z: (%.7f, %.7f, %.7f)\n", Z[0], Z[1], Z[2]);
    sprintf(strC0, "colors0: (%.7f, %.7f, %.7f)\n", colors[0][0], colors[0][1], colors[0][2]);
    sprintf(strC1, "colors1: (%.7f, %.7f, %.7f)\n", colors[1][0], colors[1][1], colors[1][2]);
    sprintf(strC2, "colors2: (%.7f, %.7f, %.7f)\n", colors[2][0], colors[2][1], colors[2][2]);
    o << strX << strY << strZ << strC0 << strC1 << strC2;
}

double
Triangle::diffuse(LightingParameters &lp, double *normal) // |<L,N>| 2-sided lighting
{
    /*
    double d = 0;
    for(int i = 0; i < 3; ++i)
    {
        cerr << "normal[" << i << "] = " << normal[i] << endl;
        cerr << "lightDir[" << i << "] = " << lp.lightDir[i] << endl;
        d += normal[i]*lp.lightDir[i];
        cerr << "d: " << d << endl;
        cin.ignore();
    }
    */
    double dif = fabs(lp.lightDir[0]*normal[0] + lp.lightDir[1]*normal[1] + lp.lightDir[2]*normal[2]);
    //cerr << "Returning: " << dif << endl;
    //cerr << "abs(d): " << fabs(d) << endl;
    return dif;
}

/*
double
Triangle::specular(LightingParameters &lp, double *normal)
{
    double r = 0; // surface reflectivity
    double s = 0;
    for(int i = 0; i < 3; ++i)
    {
        r = 2*(lp.lightDir[i]*normal[i])*normal[i] - 
    }
    double spec = lp.ks*pow(,);
}
*/

double
Triangle::calculatePhongShading(LightingParameters &lp,
                                double *vert,
                                Camera &c, // viewing direction
                                double *n) // Normal
{

    /*
     *      Do our shading
     */
    // Get diffuse 
    double dif = 0;
    for(int i = 0; i < 3; ++i)
        dif += lp.lightDir[i]*n[i];
    dif = fabs(dif);

    // Specular
    double r[3] = {0,0,0};
    double innerProd = 0;
    double rNorm = 0;
    for(int i = 0; i < 3; ++i)
        innerProd += lp.lightDir[i] * n[i];
    //cerr << "<l,n> = " << innerProd << endl;
    innerProd *= 2;
    for(int i = 0; i < 3; ++i)
        r[i] = innerProd*n[i] - lp.lightDir[i];
    double view_dir[3] = { c.position[0] - vert[0],
                           c.position[1] - vert[1],
                           c.position[2] - vert[2]};
    double vd_norm = 0;
    for(int i = 0; i < 3; ++i)
        vd_norm += (view_dir[i]*view_dir[i]);
    vd_norm = sqrt(vd_norm);
    for(int i = 0; i < 3; ++i)
        view_dir[i] /= vd_norm;
    double spec = 0;
    // <R,V>
    for(int i = 0; i < 3; ++i)
        spec += r[i]*view_dir[i];
    if(spec < 0) spec = 0;
    spec = pow(spec,lp.alpha);


    /* 
     *      Does it work?
     *      [x] Ambient
     *      [~] Diffuse
     *          - Many are off by one pixel on only one color
     *          - First 10 triangles match
     *      [~] Specular
     *          [x] R's match
     *          [x] Verts  match
     *          [x] Norms match
     *          - Has spec > 0 @ correct tris and points, but numbers are different
     */
    //cerr << "Ambient: " << lp.Ka << endl;
    //cerr << "Diffuse: " << lp.Kd*dif << endl;
    //cerr << "Specula: " << lp.Ks*spec << endl;
    //double phong = lp.Kd*dif;
    //double phong = lp.Ks*spec;
    double phong = lp.Ka + (lp.Kd*dif) + lp.Ks*spec;
    //cerr << "Phong: " << phong << endl;
    //cin.ignore();
    return phong;
}

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1e_geometry.vtk");
    //cerr << "Reading" << endl;
    rdr->Update();
    //cerr << "Done reading" << endl;
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();

    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkDoubleArray *var = (vtkDoubleArray *) pd->GetPointData()->GetArray("hardyglobal");
    double *color_ptr = var->GetPointer(0);
    //vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    //float *color_ptr = var->GetPointer(0);
    vtkFloatArray *n = (vtkFloatArray *) pd->GetPointData()->GetNormals();
    float *normals = n->GetPointer(0);
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
        double *pt = NULL;
        pt = pts->GetPoint(ptIds[0]);
        tris[idx].X[0] = pt[0];
        tris[idx].Y[0] = pt[1];
        tris[idx].Z[0] = pt[2];
#ifdef NORMALS
        tris[idx].normals[0][0] = normals[3*ptIds[0]+0];
        tris[idx].normals[0][1] = normals[3*ptIds[0]+1];
        tris[idx].normals[0][2] = normals[3*ptIds[0]+2];
#endif
        pt = pts->GetPoint(ptIds[1]);
        tris[idx].X[1] = pt[0];
        tris[idx].Y[1] = pt[1];
        tris[idx].Z[1] = pt[2];
#ifdef NORMALS
        tris[idx].normals[1][0] = normals[3*ptIds[1]+0];
        tris[idx].normals[1][1] = normals[3*ptIds[1]+1];
        tris[idx].normals[1][2] = normals[3*ptIds[1]+2];
#endif
        pt = pts->GetPoint(ptIds[2]);
        tris[idx].X[2] = pt[0];
        tris[idx].Y[2] = pt[1];
        tris[idx].Z[2] = pt[2];
#ifdef NORMALS
        tris[idx].normals[2][0] = normals[3*ptIds[2]+0];
        tris[idx].normals[2][1] = normals[3*ptIds[2]+1];
        tris[idx].normals[2][2] = normals[3*ptIds[2]+2];
#endif

        // 1->2 interpolate between light blue, dark blue
        // 2->2.5 interpolate between dark blue, cyan
        // 2.5->3 interpolate between cyan, green
        // 3->3.5 interpolate between green, yellow
        // 3.5->4 interpolate between yellow, orange
        // 4->5 interpolate between orange, brick
        // 5->6 interpolate between brick, salmon
        double mins[7] = { 1, 2, 2.5, 3, 3.5, 4, 5 };
        double maxs[7] = { 2, 2.5, 3, 3.5, 4, 5, 6 };
        unsigned char RGB[8][3] = { { 71, 71, 219 }, 
                                    { 0, 0, 91 },
                                    { 0, 255, 255 },
                                    { 0, 128, 0 },
                                    { 255, 255, 0 },
                                    { 255, 96, 0 },
                                    { 107, 0, 0 },
                                    { 224, 76, 76 } 
                                  };
        for (int j = 0 ; j < 3 ; j++)
        {
            float val = color_ptr[ptIds[j]];
            int r;
            for (r = 0 ; r < 7 ; r++)
            {
                if (mins[r] <= val && val < maxs[r])
                    break;
            }
            if (r == 7)
            {
                cerr << "Could not interpolate color for " << val << endl;
                exit(EXIT_FAILURE);
            }
            double proportion = (val-mins[r]) / (maxs[r]-mins[r]);
            tris[idx].colors[j][0] = (RGB[r][0]+proportion*(RGB[r+1][0]-RGB[r][0]))/255.0;
            tris[idx].colors[j][1] = (RGB[r][1]+proportion*(RGB[r+1][1]-RGB[r][1]))/255.0;
            tris[idx].colors[j][2] = (RGB[r][2]+proportion*(RGB[r+1][2]-RGB[r][2]))/255.0;
        }
    }

    return tris;
}

vtkImageData*
initializeScreen()
{
   vtkImageData *image = NewImage(Nx, Ny);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   int npixels = Nx*Ny;
   // Initialize everything as black
   for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;
   double zbuffer[npixels];
   for(int i = 0; i < npixels; ++i)
       zbuffer[i] = -1;
   screen.buffer  = buffer;
   screen.zbuffer = zbuffer;

   return image;
}


std::vector<Triangle>
normalize(std::vector<Triangle> t)
{
    std::vector<Triangle> nt(t.size());
    for(int i = 0; i < t.size(); ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            double magnitude = sqrt(t[i].X[j]*t[i].X[j] 
                                    + t[i].Y[j]*t[i].Y[j] 
                                    + t[i].Z[j]*t[i].Z[j]);
            nt[i].X[j] = t[i].X[j] / magnitude;
            nt[i].Y[j] = t[i].Y[j] / magnitude;
            nt[i].Z[j] = t[i].Z[j] / magnitude;
        }
    }
    return nt;
}


void
rotateAndRender(Matrix &m, Triangle t)
{
    t.applyShading();
    for(int i = 0; i < 3; ++i)
    {
        double pIn[4] = {t.X[i], t.Y[i], t.Z[i], 1};
        double pOt[4] = {0,0,0,0};
        m.TransformPoint(pIn, pOt);
        t.X[i] = pOt[0]/pOt[3];
        t.Y[i] = pOt[1]/pOt[3];
        t.Z[i] = pOt[2]/pOt[3];
    }


    t.raster();
}


int writeScreen(int c0, int c1, char* s)
{
   vtkImageData *image = initializeScreen();
   Camera camera = GetCamera(c0,c1);

   std::vector<Triangle> triangles = GetTriangles();
   Matrix m = camera.VT_CT_DT();
   for(int i = 0; i < triangles.size(); ++i)
   {
       //cerr << "Triangle: " << i << endl;
       triangles[i].setCamera(camera);
       rotateAndRender(m, triangles[i]);
   }
   WriteImage(image, s);
}


int main()
{
    //for(int i = 0; i < 1000; ++i)
    for(int i = 0; i < 1; ++i)
    {
        char fname[256];
        sprintf(fname, "frame%.3i",i);
        cerr << "Frame: " << fname << endl;
        writeScreen(i,1000, fname);
    }
    return 0;
}
