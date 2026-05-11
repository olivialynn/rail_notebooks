GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4537721e-01	 3.2113150e-01	-3.3505362e-01	 3.1970793e-01	[-3.3092947e-01]	 4.5179009e-01


.. parsed-literal::

       2	-2.7511596e-01	 3.1046669e-01	-2.5047460e-01	 3.1183015e-01	[-2.4728639e-01]	 2.2238374e-01


.. parsed-literal::

       3	-2.2797106e-01	 2.8777508e-01	-1.8252789e-01	 2.9129045e-01	[-1.8805132e-01]	 2.8580523e-01


.. parsed-literal::

       4	-2.0062554e-01	 2.6478835e-01	-1.5947838e-01	 2.6953947e-01	[-1.7968406e-01]	 2.0620632e-01
       5	-1.0289166e-01	 2.5725776e-01	-6.7601387e-02	 2.6279532e-01	[-8.0144070e-02]	 1.9496584e-01


.. parsed-literal::

       6	-7.2750645e-02	 2.5203650e-01	-4.1472930e-02	 2.5350837e-01	[-4.3650664e-02]	 2.0417571e-01
       7	-5.3481236e-02	 2.4909144e-01	-2.9564653e-02	 2.4946453e-01	[-2.9014067e-02]	 1.9528770e-01


.. parsed-literal::

       8	-4.2091225e-02	 2.4718773e-01	-2.1584042e-02	 2.4744254e-01	[-2.0433856e-02]	 1.7612624e-01


.. parsed-literal::

       9	-2.8867348e-02	 2.4471889e-01	-1.1019420e-02	 2.4483546e-01	[-9.7353752e-03]	 2.1686006e-01


.. parsed-literal::

      10	-1.8675002e-02	 2.4282716e-01	-2.6276686e-03	 2.4217846e-01	[ 1.1908036e-03]	 2.0793581e-01
      11	-1.4162919e-02	 2.4214847e-01	 7.8748654e-04	 2.4125903e-01	[ 5.5651505e-03]	 1.9387507e-01


.. parsed-literal::

      12	-1.1077286e-02	 2.4144785e-01	 3.7142469e-03	 2.4039805e-01	[ 9.4399575e-03]	 2.0978642e-01


.. parsed-literal::

      13	-6.5839940e-03	 2.4050186e-01	 8.2933757e-03	 2.3945497e-01	[ 1.4759124e-02]	 2.0724630e-01


.. parsed-literal::

      14	 1.2380313e-01	 2.2453678e-01	 1.4684346e-01	 2.3042991e-01	[ 1.5743269e-01]	 4.3159056e-01


.. parsed-literal::

      15	 1.5100564e-01	 2.2445434e-01	 1.7988346e-01	 2.2241959e-01	[ 1.9484942e-01]	 2.0727944e-01
      16	 2.7549673e-01	 2.1634475e-01	 3.0866880e-01	 2.1560217e-01	[ 3.3427745e-01]	 1.9414210e-01


.. parsed-literal::

      17	 3.2775745e-01	 2.0939057e-01	 3.6189346e-01	 2.1001639e-01	[ 3.8883230e-01]	 2.0425296e-01
      18	 3.9727670e-01	 2.0435570e-01	 4.3297849e-01	 2.0869632e-01	[ 4.4889326e-01]	 2.0125699e-01


.. parsed-literal::

      19	 5.0134735e-01	 1.9982543e-01	 5.3908796e-01	 2.0080204e-01	[ 5.5473785e-01]	 1.9715476e-01
      20	 5.9354376e-01	 1.9949771e-01	 6.3348763e-01	 1.9784626e-01	[ 6.2696706e-01]	 1.8386197e-01


.. parsed-literal::

      21	 6.3559547e-01	 1.9764617e-01	 6.7658437e-01	 1.9696307e-01	[ 6.5528226e-01]	 1.9932103e-01
      22	 6.6815228e-01	 1.9728936e-01	 7.0830282e-01	 1.9721978e-01	[ 6.8953536e-01]	 1.9906998e-01


.. parsed-literal::

      23	 6.9813439e-01	 1.9850036e-01	 7.3833580e-01	 1.9952466e-01	[ 7.2525267e-01]	 2.1385741e-01
      24	 7.2551231e-01	 2.0845463e-01	 7.6543091e-01	 2.1206874e-01	[ 7.6712369e-01]	 1.8944311e-01


.. parsed-literal::

      25	 7.5399250e-01	 2.0941713e-01	 7.9437433e-01	 2.1376636e-01	[ 7.9852523e-01]	 1.9779372e-01
      26	 7.7552179e-01	 2.0439005e-01	 8.1707496e-01	 2.0865783e-01	[ 8.1416343e-01]	 1.9338226e-01


.. parsed-literal::

      27	 8.0933791e-01	 2.0120465e-01	 8.5248909e-01	 2.0567314e-01	[ 8.4402020e-01]	 1.9600010e-01
      28	 8.3442852e-01	 2.0117238e-01	 8.7819950e-01	 2.0461306e-01	[ 8.6519202e-01]	 1.9583702e-01


.. parsed-literal::

      29	 8.5785797e-01	 1.9957214e-01	 9.0177066e-01	 2.0202062e-01	[ 8.9041442e-01]	 1.9962621e-01
      30	 8.7645889e-01	 1.9729402e-01	 9.2032802e-01	 2.0159175e-01	[ 9.0365763e-01]	 1.9137526e-01


.. parsed-literal::

      31	 8.9269939e-01	 1.9441634e-01	 9.3679898e-01	 1.9993568e-01	[ 9.2085055e-01]	 1.9783711e-01
      32	 9.1958292e-01	 1.8818658e-01	 9.6466834e-01	 1.9602818e-01	[ 9.4927487e-01]	 1.9600487e-01


.. parsed-literal::

      33	 9.3869445e-01	 1.8437845e-01	 9.8437985e-01	 1.9156092e-01	[ 9.6304871e-01]	 1.9688964e-01
      34	 9.5765519e-01	 1.8245662e-01	 1.0041328e+00	 1.8896293e-01	[ 9.7636811e-01]	 1.9530869e-01


.. parsed-literal::

      35	 9.7264671e-01	 1.8058822e-01	 1.0197224e+00	 1.8806185e-01	[ 9.9020948e-01]	 1.9785976e-01
      36	 9.8949835e-01	 1.7906987e-01	 1.0377504e+00	 1.8726264e-01	[ 9.9534296e-01]	 1.9420218e-01


.. parsed-literal::

      37	 9.9984206e-01	 1.7808413e-01	 1.0485630e+00	 1.8671290e-01	[ 9.9912789e-01]	 2.0237875e-01


.. parsed-literal::

      38	 1.0091364e+00	 1.7781902e-01	 1.0577574e+00	 1.8616408e-01	[ 1.0035132e+00]	 2.0456862e-01
      39	 1.0229285e+00	 1.7723781e-01	 1.0718229e+00	 1.8527649e-01	[ 1.0095784e+00]	 1.9484735e-01


.. parsed-literal::

      40	 1.0358844e+00	 1.7635714e-01	 1.0851033e+00	 1.8458510e-01	[ 1.0178840e+00]	 1.9544411e-01


.. parsed-literal::

      41	 1.0487534e+00	 1.7387645e-01	 1.0980890e+00	 1.8248027e-01	[ 1.0343204e+00]	 2.0281601e-01
      42	 1.0578587e+00	 1.7232166e-01	 1.1070741e+00	 1.8069885e-01	[ 1.0417489e+00]	 1.9454837e-01


.. parsed-literal::

      43	 1.0687178e+00	 1.6939964e-01	 1.1180609e+00	 1.7709913e-01	[ 1.0524088e+00]	 1.9009805e-01
      44	 1.0807053e+00	 1.6518277e-01	 1.1305657e+00	 1.7204268e-01	[ 1.0630335e+00]	 2.0237327e-01


.. parsed-literal::

      45	 1.0929032e+00	 1.6101734e-01	 1.1432751e+00	 1.6622926e-01	[ 1.0755336e+00]	 2.0521140e-01
      46	 1.1027088e+00	 1.5932033e-01	 1.1530998e+00	 1.6408095e-01	[ 1.0843234e+00]	 1.9795990e-01


.. parsed-literal::

      47	 1.1183000e+00	 1.5695744e-01	 1.1689904e+00	 1.6212407e-01	[ 1.1006165e+00]	 2.0271850e-01


.. parsed-literal::

      48	 1.1266613e+00	 1.5577579e-01	 1.1776702e+00	 1.6189184e-01	  1.0959395e+00 	 2.0403361e-01
      49	 1.1358105e+00	 1.5455451e-01	 1.1867093e+00	 1.6111695e-01	[ 1.1093340e+00]	 1.9130421e-01


.. parsed-literal::

      50	 1.1490904e+00	 1.5221374e-01	 1.2003262e+00	 1.5952790e-01	[ 1.1260754e+00]	 2.1114302e-01
      51	 1.1598522e+00	 1.5077076e-01	 1.2113254e+00	 1.5890093e-01	[ 1.1368485e+00]	 2.0190549e-01


.. parsed-literal::

      52	 1.1734192e+00	 1.4851832e-01	 1.2255646e+00	 1.5869557e-01	[ 1.1460201e+00]	 2.0282292e-01


.. parsed-literal::

      53	 1.1839997e+00	 1.4752795e-01	 1.2361644e+00	 1.5736779e-01	[ 1.1524076e+00]	 2.1209407e-01


.. parsed-literal::

      54	 1.1911116e+00	 1.4708903e-01	 1.2431821e+00	 1.5700096e-01	[ 1.1525185e+00]	 2.0402098e-01


.. parsed-literal::

      55	 1.1974707e+00	 1.4526496e-01	 1.2505012e+00	 1.5350021e-01	[ 1.1535243e+00]	 2.0744586e-01


.. parsed-literal::

      56	 1.2100055e+00	 1.4550869e-01	 1.2626774e+00	 1.5456939e-01	[ 1.1562380e+00]	 2.0447564e-01


.. parsed-literal::

      57	 1.2158011e+00	 1.4472456e-01	 1.2684595e+00	 1.5438513e-01	[ 1.1593641e+00]	 2.0386958e-01
      58	 1.2230543e+00	 1.4392893e-01	 1.2760711e+00	 1.5430067e-01	  1.1578264e+00 	 1.7530990e-01


.. parsed-literal::

      59	 1.2331807e+00	 1.4271483e-01	 1.2867631e+00	 1.5358642e-01	  1.1519696e+00 	 2.1086788e-01
      60	 1.2400313e+00	 1.4261977e-01	 1.2943705e+00	 1.5468009e-01	  1.1455976e+00 	 1.9073200e-01


.. parsed-literal::

      61	 1.2507178e+00	 1.4164605e-01	 1.3046974e+00	 1.5354919e-01	  1.1523490e+00 	 1.8038464e-01
      62	 1.2577354e+00	 1.4101879e-01	 1.3114176e+00	 1.5269120e-01	[ 1.1594950e+00]	 1.9724154e-01


.. parsed-literal::

      63	 1.2659151e+00	 1.4051221e-01	 1.3195107e+00	 1.5214511e-01	[ 1.1726162e+00]	 1.9701958e-01
      64	 1.2724860e+00	 1.3978616e-01	 1.3264532e+00	 1.5163735e-01	[ 1.1792445e+00]	 1.6771054e-01


.. parsed-literal::

      65	 1.2821500e+00	 1.3956050e-01	 1.3358978e+00	 1.5177628e-01	[ 1.1988223e+00]	 1.9890523e-01
      66	 1.2870367e+00	 1.3931980e-01	 1.3407323e+00	 1.5193901e-01	[ 1.2049331e+00]	 1.8581724e-01


.. parsed-literal::

      67	 1.2942336e+00	 1.3880229e-01	 1.3480928e+00	 1.5191536e-01	[ 1.2125935e+00]	 1.8397403e-01


.. parsed-literal::

      68	 1.3006228e+00	 1.3825633e-01	 1.3546034e+00	 1.5150865e-01	[ 1.2206521e+00]	 2.0100379e-01
      69	 1.3086048e+00	 1.3767011e-01	 1.3626394e+00	 1.5150044e-01	[ 1.2270397e+00]	 1.9581628e-01


.. parsed-literal::

      70	 1.3145007e+00	 1.3705408e-01	 1.3685832e+00	 1.5119403e-01	[ 1.2282776e+00]	 1.9484591e-01
      71	 1.3224723e+00	 1.3623665e-01	 1.3768174e+00	 1.5042171e-01	  1.2252667e+00 	 1.9592094e-01


.. parsed-literal::

      72	 1.3325958e+00	 1.3485328e-01	 1.3875651e+00	 1.4886413e-01	  1.2154007e+00 	 1.9775844e-01


.. parsed-literal::

      73	 1.3360087e+00	 1.3455275e-01	 1.3915966e+00	 1.4850972e-01	  1.1990208e+00 	 2.1109533e-01


.. parsed-literal::

      74	 1.3440383e+00	 1.3425876e-01	 1.3992084e+00	 1.4844663e-01	  1.2140292e+00 	 2.0373583e-01
      75	 1.3480881e+00	 1.3396074e-01	 1.4032764e+00	 1.4836715e-01	  1.2172775e+00 	 1.7763972e-01


.. parsed-literal::

      76	 1.3557635e+00	 1.3318407e-01	 1.4110934e+00	 1.4829415e-01	  1.2193272e+00 	 2.0704460e-01
      77	 1.3574588e+00	 1.3255597e-01	 1.4131786e+00	 1.4828058e-01	  1.2194808e+00 	 2.0252943e-01


.. parsed-literal::

      78	 1.3664584e+00	 1.3195355e-01	 1.4219984e+00	 1.4803399e-01	  1.2276495e+00 	 1.9651890e-01
      79	 1.3696287e+00	 1.3172266e-01	 1.4251125e+00	 1.4783963e-01	[ 1.2296796e+00]	 1.8609619e-01


.. parsed-literal::

      80	 1.3742027e+00	 1.3118677e-01	 1.4297764e+00	 1.4752642e-01	[ 1.2301657e+00]	 1.8487167e-01
      81	 1.3788204e+00	 1.3063895e-01	 1.4345246e+00	 1.4664442e-01	[ 1.2333007e+00]	 1.9652581e-01


.. parsed-literal::

      82	 1.3838893e+00	 1.3010401e-01	 1.4395724e+00	 1.4618580e-01	[ 1.2372439e+00]	 1.9391799e-01
      83	 1.3879361e+00	 1.2962141e-01	 1.4436134e+00	 1.4564065e-01	[ 1.2431241e+00]	 1.9647026e-01


.. parsed-literal::

      84	 1.3926123e+00	 1.2901879e-01	 1.4482880e+00	 1.4516769e-01	[ 1.2484171e+00]	 2.0285654e-01


.. parsed-literal::

      85	 1.3963303e+00	 1.2838275e-01	 1.4521945e+00	 1.4437535e-01	[ 1.2576209e+00]	 2.1088529e-01


.. parsed-literal::

      86	 1.4022120e+00	 1.2803875e-01	 1.4578938e+00	 1.4448568e-01	[ 1.2613226e+00]	 2.0531297e-01
      87	 1.4052429e+00	 1.2786584e-01	 1.4608628e+00	 1.4462865e-01	[ 1.2625484e+00]	 2.0093966e-01


.. parsed-literal::

      88	 1.4097323e+00	 1.2754099e-01	 1.4654249e+00	 1.4456544e-01	[ 1.2639540e+00]	 1.9681144e-01
      89	 1.4138243e+00	 1.2689856e-01	 1.4697597e+00	 1.4396793e-01	  1.2621645e+00 	 1.8369699e-01


.. parsed-literal::

      90	 1.4189935e+00	 1.2678265e-01	 1.4750817e+00	 1.4375599e-01	[ 1.2642610e+00]	 1.8819356e-01
      91	 1.4213644e+00	 1.2663786e-01	 1.4775052e+00	 1.4342764e-01	[ 1.2659757e+00]	 1.9788742e-01


.. parsed-literal::

      92	 1.4247782e+00	 1.2630381e-01	 1.4810871e+00	 1.4307616e-01	  1.2631419e+00 	 1.8818450e-01
      93	 1.4282641e+00	 1.2606646e-01	 1.4848932e+00	 1.4256256e-01	  1.2600149e+00 	 1.8615508e-01


.. parsed-literal::

      94	 1.4325191e+00	 1.2577817e-01	 1.4891310e+00	 1.4283276e-01	  1.2553406e+00 	 1.9406176e-01
      95	 1.4356979e+00	 1.2565119e-01	 1.4922714e+00	 1.4332567e-01	  1.2534964e+00 	 1.9463611e-01


.. parsed-literal::

      96	 1.4383193e+00	 1.2561552e-01	 1.4947942e+00	 1.4353918e-01	  1.2569988e+00 	 1.8155146e-01
      97	 1.4420758e+00	 1.2550886e-01	 1.4984835e+00	 1.4396587e-01	  1.2636356e+00 	 1.8917108e-01


.. parsed-literal::

      98	 1.4455609e+00	 1.2546552e-01	 1.5019418e+00	 1.4374430e-01	[ 1.2700850e+00]	 1.9887233e-01


.. parsed-literal::

      99	 1.4489807e+00	 1.2523141e-01	 1.5054092e+00	 1.4342420e-01	[ 1.2767886e+00]	 2.0998049e-01


.. parsed-literal::

     100	 1.4530020e+00	 1.2481674e-01	 1.5097144e+00	 1.4292224e-01	[ 1.2783914e+00]	 2.1877551e-01
     101	 1.4560223e+00	 1.2441287e-01	 1.5128038e+00	 1.4244952e-01	[ 1.2860737e+00]	 1.8097544e-01


.. parsed-literal::

     102	 1.4587355e+00	 1.2414459e-01	 1.5154731e+00	 1.4232822e-01	[ 1.2861901e+00]	 1.7779756e-01
     103	 1.4620481e+00	 1.2385314e-01	 1.5187876e+00	 1.4225433e-01	[ 1.2870621e+00]	 1.9729877e-01


.. parsed-literal::

     104	 1.4646050e+00	 1.2365297e-01	 1.5212475e+00	 1.4192608e-01	[ 1.2909400e+00]	 1.8856812e-01


.. parsed-literal::

     105	 1.4663336e+00	 1.2360666e-01	 1.5229447e+00	 1.4180503e-01	[ 1.2944496e+00]	 2.0411158e-01


.. parsed-literal::

     106	 1.4706243e+00	 1.2353652e-01	 1.5272965e+00	 1.4158324e-01	[ 1.2996968e+00]	 2.0230269e-01
     107	 1.4743944e+00	 1.2331820e-01	 1.5312518e+00	 1.4089135e-01	[ 1.3011096e+00]	 1.9728160e-01


.. parsed-literal::

     108	 1.4778146e+00	 1.2295019e-01	 1.5348713e+00	 1.4058370e-01	  1.2941426e+00 	 2.0586896e-01
     109	 1.4799640e+00	 1.2283640e-01	 1.5370419e+00	 1.4034507e-01	  1.2962026e+00 	 1.9142723e-01


.. parsed-literal::

     110	 1.4814987e+00	 1.2264355e-01	 1.5385260e+00	 1.4019256e-01	  1.2953742e+00 	 2.1098399e-01
     111	 1.4834168e+00	 1.2248626e-01	 1.5404281e+00	 1.4011800e-01	  1.2954820e+00 	 1.8218446e-01


.. parsed-literal::

     112	 1.4865403e+00	 1.2230418e-01	 1.5435940e+00	 1.4024063e-01	  1.2955559e+00 	 2.0360374e-01


.. parsed-literal::

     113	 1.4883005e+00	 1.2230738e-01	 1.5454970e+00	 1.4024326e-01	  1.2982147e+00 	 2.1386504e-01


.. parsed-literal::

     114	 1.4905472e+00	 1.2231840e-01	 1.5476659e+00	 1.4048194e-01	  1.2995333e+00 	 2.1451068e-01
     115	 1.4930708e+00	 1.2240782e-01	 1.5501554e+00	 1.4077199e-01	  1.3008536e+00 	 2.0069146e-01


.. parsed-literal::

     116	 1.4954930e+00	 1.2243442e-01	 1.5526197e+00	 1.4102757e-01	  1.3001651e+00 	 2.1004891e-01


.. parsed-literal::

     117	 1.4974851e+00	 1.2262014e-01	 1.5548239e+00	 1.4116763e-01	  1.3004785e+00 	 2.1536183e-01


.. parsed-literal::

     118	 1.5002711e+00	 1.2246106e-01	 1.5575526e+00	 1.4118206e-01	  1.2996368e+00 	 2.0590901e-01
     119	 1.5016701e+00	 1.2224903e-01	 1.5589455e+00	 1.4097409e-01	  1.2995394e+00 	 1.9619560e-01


.. parsed-literal::

     120	 1.5036558e+00	 1.2199997e-01	 1.5609727e+00	 1.4077110e-01	  1.2981273e+00 	 2.0905495e-01
     121	 1.5041947e+00	 1.2168390e-01	 1.5616990e+00	 1.4040283e-01	  1.3000420e+00 	 2.0038891e-01


.. parsed-literal::

     122	 1.5065657e+00	 1.2164370e-01	 1.5639531e+00	 1.4053095e-01	  1.2983184e+00 	 1.9667125e-01
     123	 1.5074780e+00	 1.2165144e-01	 1.5648562e+00	 1.4062270e-01	  1.2985899e+00 	 1.9100833e-01


.. parsed-literal::

     124	 1.5090495e+00	 1.2156996e-01	 1.5664623e+00	 1.4067869e-01	  1.2989616e+00 	 1.9668126e-01
     125	 1.5113761e+00	 1.2143644e-01	 1.5688650e+00	 1.4068746e-01	  1.2991103e+00 	 1.9888210e-01


.. parsed-literal::

     126	 1.5121748e+00	 1.2085170e-01	 1.5700352e+00	 1.4076631e-01	  1.2884028e+00 	 2.0050621e-01
     127	 1.5156303e+00	 1.2101320e-01	 1.5732483e+00	 1.4069204e-01	  1.2951852e+00 	 2.0413232e-01


.. parsed-literal::

     128	 1.5166691e+00	 1.2096789e-01	 1.5742461e+00	 1.4063184e-01	  1.2950060e+00 	 2.0950198e-01
     129	 1.5187367e+00	 1.2078359e-01	 1.5763160e+00	 1.4061196e-01	  1.2920845e+00 	 2.0179248e-01


.. parsed-literal::

     130	 1.5206095e+00	 1.2042514e-01	 1.5782691e+00	 1.4043655e-01	  1.2882910e+00 	 1.7061877e-01


.. parsed-literal::

     131	 1.5231111e+00	 1.2015507e-01	 1.5807702e+00	 1.4042867e-01	  1.2845184e+00 	 2.0524597e-01


.. parsed-literal::

     132	 1.5248554e+00	 1.1992992e-01	 1.5825608e+00	 1.4040932e-01	  1.2818078e+00 	 2.0642996e-01


.. parsed-literal::

     133	 1.5260384e+00	 1.1969989e-01	 1.5838565e+00	 1.4021711e-01	  1.2819067e+00 	 2.0217443e-01
     134	 1.5273300e+00	 1.1955392e-01	 1.5852066e+00	 1.4015260e-01	  1.2791419e+00 	 2.0648146e-01


.. parsed-literal::

     135	 1.5288105e+00	 1.1941136e-01	 1.5867462e+00	 1.3999881e-01	  1.2768436e+00 	 1.9626093e-01


.. parsed-literal::

     136	 1.5309788e+00	 1.1917726e-01	 1.5890374e+00	 1.3983666e-01	  1.2702183e+00 	 2.1318316e-01


.. parsed-literal::

     137	 1.5318224e+00	 1.1909941e-01	 1.5899155e+00	 1.3983321e-01	  1.2685826e+00 	 3.2313657e-01
     138	 1.5332678e+00	 1.1900023e-01	 1.5913885e+00	 1.3985250e-01	  1.2637613e+00 	 1.8376803e-01


.. parsed-literal::

     139	 1.5348286e+00	 1.1894237e-01	 1.5929646e+00	 1.3995887e-01	  1.2592546e+00 	 1.8430448e-01
     140	 1.5363545e+00	 1.1889775e-01	 1.5944713e+00	 1.4011972e-01	  1.2549279e+00 	 1.7193580e-01


.. parsed-literal::

     141	 1.5376949e+00	 1.1906017e-01	 1.5958426e+00	 1.4023317e-01	  1.2549592e+00 	 1.7997909e-01
     142	 1.5391423e+00	 1.1900761e-01	 1.5972171e+00	 1.4027951e-01	  1.2536305e+00 	 2.0235777e-01


.. parsed-literal::

     143	 1.5400508e+00	 1.1901178e-01	 1.5980910e+00	 1.4018083e-01	  1.2557802e+00 	 2.0172215e-01
     144	 1.5412956e+00	 1.1897029e-01	 1.5993477e+00	 1.3998959e-01	  1.2553933e+00 	 1.9375134e-01


.. parsed-literal::

     145	 1.5427022e+00	 1.1889450e-01	 1.6008559e+00	 1.3966293e-01	  1.2487148e+00 	 2.0154333e-01


.. parsed-literal::

     146	 1.5443869e+00	 1.1871285e-01	 1.6025662e+00	 1.3948658e-01	  1.2437085e+00 	 2.0973659e-01


.. parsed-literal::

     147	 1.5456795e+00	 1.1855968e-01	 1.6038763e+00	 1.3938093e-01	  1.2398851e+00 	 2.1586108e-01
     148	 1.5469258e+00	 1.1835203e-01	 1.6051725e+00	 1.3932765e-01	  1.2327879e+00 	 1.9855142e-01


.. parsed-literal::

     149	 1.5481091e+00	 1.1816102e-01	 1.6063983e+00	 1.3926354e-01	  1.2285305e+00 	 2.0930982e-01
     150	 1.5496082e+00	 1.1805697e-01	 1.6078674e+00	 1.3920456e-01	  1.2280223e+00 	 1.9722199e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 1s, sys: 911 ms, total: 2min 2s
    Wall time: 30.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2278a8f3a0>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 975 ms, sys: 43.9 ms, total: 1.02 s
    Wall time: 381 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

