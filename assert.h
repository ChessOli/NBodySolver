#ifndef ASSERT_GASPI_H
#define ASSERT_GASPI_H

#include <stdio.h>
#include <stdlib.h>

#define ASSERT(x...)                                                    \
  if (!(x))                                                             \
  {                                                                     \
    fprintf (stderr, "Error: '%s' [%s:%i]\n", #x, __FILE__, __LINE__);  \
    exit (EXIT_FAILURE);                                                \
  }

#endif
