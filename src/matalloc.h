/*
 * Copyright (c) 2013 Egor Zindy
 *
 * matalloc is free software: you can redistribute it and/or modify  
 * it under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef struct
{
    int l;      //l matrices
    int m;      //m rows
    int n;      //n columns

    void *data; //data block
    void **me;  //pointers to first elem of each row
    void ***te; //pointers to first elem of each matrix

    size_t dsize;  //size of the elements in bytes
    int is_managed; //are we wrapping an existing block of memory (is_managed=1) or memory was allocated as part of the creation (is_managed=0)

} tensor;

typedef struct
{
    int m;      //m rows
    int n;      //n columns

    void *data; //data block
    void **me;  //pointers to first elem of each row

    size_t dsize;  //size of the elements in bytes
    int is_managed; //when calling m_free(), do we free data? Yes if is_manged=1, no otherwise.

} matrix;

typedef struct
{
    int dim;    //n elements
    void *ve;  //data block

    size_t dsize;  //size of the elements in bytes
    int is_managed; //when calling v_free(), do we free ve? Yes if is_manged=1, no otherwise.

} vector;

vector *v_copy(vector *vec1, vector *vecOut);
vector *v_wrap(void *data, int dim, size_t dsize);
vector *v_get(int dim, size_t dsize);
void v_info(vector *vec1);
void v_free(vector *vec);

tensor *t_copy(tensor *ten1, tensor *tenOut);
tensor *t_wrap2(void **data, int l,int m, int n, size_t dsize);
tensor *t_get2(int l, int m, int n, size_t dsize);
tensor *t_wrap(void *data, int l,int m, int n, size_t dsize);
tensor *t_get(int l, int m, int n, size_t dsize);
void t_info(tensor *ten1);
void t_free(tensor *ten);

matrix *m_copy(matrix *mat1, matrix *matOut);
matrix *m_wrap(void *data, int m, int n, size_t dsize);
matrix *m_get(int m, int n, size_t dsize);
void m_info(matrix *mat1);
void m_free(matrix *mat);

