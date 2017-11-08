// C++ 17 Includes:
#include <iostream>

// Third Party Includes:

// Project Includes:
#include "globalmem.cuh"
#include "sharedmem.cuh"

int main(int argc, char** argv)
{
  try
  {
    demoStaticGlobal();
    demoDynamicGlobal();
    demoDynamicShared();
    std::cout << "All good.\n";
  }
  catch (const std::runtime_error& e)
  {
    std::cerr << "Demo failed:  " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
