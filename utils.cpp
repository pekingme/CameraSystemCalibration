#include "utils.h"

bool Utils::fileExists ( const string& filename )
{
    if ( FILE* file = fopen ( filename.c_str(), "r" ) )
    {
        fclose ( file );
        return true;
    }
    else
    {
        cerr << "File not found: " << filename << endl;
        return false;
    }
}
