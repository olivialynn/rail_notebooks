GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

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
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

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
       1	-3.5110293e-01	 3.2291679e-01	-3.4147390e-01	 3.1220696e-01	[-3.2099334e-01]	 4.6050024e-01


.. parsed-literal::

       2	-2.8205036e-01	 3.1326406e-01	-2.5922999e-01	 3.0080271e-01	[-2.2280289e-01]	 2.2868419e-01


.. parsed-literal::

       3	-2.3570489e-01	 2.9115733e-01	-1.9344496e-01	 2.7897744e-01	[-1.4227263e-01]	 2.7811813e-01


.. parsed-literal::

       4	-2.0289579e-01	 2.6838037e-01	-1.6140404e-01	 2.6057015e-01	[-1.1070554e-01]	 2.0415950e-01


.. parsed-literal::

       5	-1.1250694e-01	 2.5991499e-01	-7.8984100e-02	 2.5170653e-01	[-3.9898606e-02]	 2.0576406e-01
       6	-8.0558755e-02	 2.5426660e-01	-5.0507325e-02	 2.4490763e-01	[-1.5609229e-02]	 1.9818187e-01


.. parsed-literal::

       7	-6.1802608e-02	 2.5131174e-01	-3.8191693e-02	 2.4114591e-01	[ 1.0221169e-03]	 2.0248437e-01


.. parsed-literal::

       8	-4.7486359e-02	 2.4876212e-01	-2.7676239e-02	 2.3872313e-01	[ 1.2302620e-02]	 2.1212840e-01


.. parsed-literal::

       9	-3.2483968e-02	 2.4585508e-01	-1.5520572e-02	 2.3701211e-01	[ 2.2117785e-02]	 2.1544981e-01


.. parsed-literal::

      10	-2.2527116e-02	 2.4406447e-01	-7.6797415e-03	 2.3677965e-01	[ 2.5743113e-02]	 2.0429921e-01
      11	-1.6919900e-02	 2.4299387e-01	-2.9329205e-03	 2.3556849e-01	[ 3.1830704e-02]	 1.8304420e-01


.. parsed-literal::

      12	-1.4499086e-02	 2.4254897e-01	-6.7981221e-04	 2.3509026e-01	[ 3.2654072e-02]	 1.8549228e-01
      13	-1.0265733e-02	 2.4179015e-01	 3.4859805e-03	 2.3395357e-01	[ 3.7770090e-02]	 1.8329930e-01


.. parsed-literal::

      14	 2.0771454e-02	 2.3311980e-01	 3.6731466e-02	 2.2412911e-01	[ 6.5981329e-02]	 2.9192424e-01


.. parsed-literal::

      15	 3.6486766e-02	 2.2886805e-01	 5.2327377e-02	 2.2072731e-01	[ 7.3464514e-02]	 3.1517076e-01
      16	 9.3370857e-02	 2.2574882e-01	 1.1284068e-01	 2.1819442e-01	[ 1.3608749e-01]	 1.7824912e-01


.. parsed-literal::

      17	 2.7461091e-01	 2.2407142e-01	 3.0639536e-01	 2.1997347e-01	[ 3.2112885e-01]	 2.1112657e-01
      18	 3.3901232e-01	 2.1837612e-01	 3.6983015e-01	 2.1102075e-01	[ 3.7557832e-01]	 1.9961357e-01


.. parsed-literal::

      19	 3.9439686e-01	 2.1298074e-01	 4.2598337e-01	 2.0388657e-01	[ 4.3464466e-01]	 1.7799854e-01


.. parsed-literal::

      20	 4.5293912e-01	 2.0834234e-01	 4.8608924e-01	 2.0010453e-01	[ 4.8906857e-01]	 2.1773672e-01
      21	 5.2376969e-01	 2.0526808e-01	 5.6009420e-01	 1.9864331e-01	[ 5.5607599e-01]	 1.7435122e-01


.. parsed-literal::

      22	 6.0122841e-01	 2.0302967e-01	 6.3937897e-01	 1.9455993e-01	[ 6.1971335e-01]	 2.0755100e-01
      23	 6.3959559e-01	 2.0102451e-01	 6.7730486e-01	 1.9182515e-01	[ 6.4996625e-01]	 1.7930698e-01


.. parsed-literal::

      24	 6.7215903e-01	 1.9723402e-01	 7.1047302e-01	 1.8811829e-01	[ 6.7546965e-01]	 1.9282198e-01


.. parsed-literal::

      25	 7.0619639e-01	 1.9620189e-01	 7.4381933e-01	 1.8581972e-01	[ 7.1441080e-01]	 2.0803785e-01


.. parsed-literal::

      26	 7.3961288e-01	 1.9582817e-01	 7.7712316e-01	 1.8576156e-01	[ 7.4498551e-01]	 2.0908332e-01
      27	 7.6924827e-01	 1.9953937e-01	 8.0721139e-01	 1.8737866e-01	[ 7.7292666e-01]	 1.7897034e-01


.. parsed-literal::

      28	 7.9989896e-01	 1.9505243e-01	 8.3910029e-01	 1.8412133e-01	[ 7.9463302e-01]	 1.7304683e-01
      29	 8.2605284e-01	 1.9439187e-01	 8.6568954e-01	 1.8390163e-01	[ 8.1384415e-01]	 1.9276905e-01


.. parsed-literal::

      30	 8.5804561e-01	 1.9333861e-01	 8.9857364e-01	 1.8017544e-01	[ 8.3743579e-01]	 1.7827916e-01


.. parsed-literal::

      31	 8.7711467e-01	 1.9283425e-01	 9.1801782e-01	 1.7920401e-01	[ 8.5611467e-01]	 2.1059322e-01


.. parsed-literal::

      32	 9.0399849e-01	 1.8735341e-01	 9.4526193e-01	 1.7528249e-01	[ 8.8188843e-01]	 2.1431208e-01
      33	 9.1949946e-01	 1.8564751e-01	 9.6100710e-01	 1.7446868e-01	[ 8.9647114e-01]	 1.7581916e-01


.. parsed-literal::

      34	 9.3802148e-01	 1.8342579e-01	 9.8017867e-01	 1.7267910e-01	[ 9.1680052e-01]	 2.1448445e-01
      35	 9.5639319e-01	 1.8130311e-01	 9.9956171e-01	 1.7170170e-01	[ 9.4083915e-01]	 1.8640661e-01


.. parsed-literal::

      36	 9.7528372e-01	 1.8030378e-01	 1.0184560e+00	 1.6958779e-01	[ 9.5932994e-01]	 2.0733500e-01
      37	 9.8693584e-01	 1.7993974e-01	 1.0303146e+00	 1.6860392e-01	[ 9.7064645e-01]	 1.9308901e-01


.. parsed-literal::

      38	 1.0108442e+00	 1.7867750e-01	 1.0557300e+00	 1.6657065e-01	[ 9.9452052e-01]	 1.9947529e-01


.. parsed-literal::

      39	 1.0236900e+00	 1.7749829e-01	 1.0689009e+00	 1.6594080e-01	[ 9.9882793e-01]	 3.2881021e-01


.. parsed-literal::

      40	 1.0342340e+00	 1.7607214e-01	 1.0802451e+00	 1.6408302e-01	[ 1.0100093e+00]	 2.0888305e-01


.. parsed-literal::

      41	 1.0445622e+00	 1.7477387e-01	 1.0910032e+00	 1.6321055e-01	[ 1.0167588e+00]	 2.0764852e-01
      42	 1.0551406e+00	 1.7333406e-01	 1.1017935e+00	 1.6201561e-01	[ 1.0266139e+00]	 2.0393848e-01


.. parsed-literal::

      43	 1.0664516e+00	 1.7146304e-01	 1.1132701e+00	 1.6078456e-01	[ 1.0380567e+00]	 2.0651221e-01
      44	 1.0784361e+00	 1.6983729e-01	 1.1249447e+00	 1.5925259e-01	[ 1.0543402e+00]	 1.9077921e-01


.. parsed-literal::

      45	 1.0852533e+00	 1.6919330e-01	 1.1317704e+00	 1.5835166e-01	[ 1.0627990e+00]	 2.0685911e-01


.. parsed-literal::

      46	 1.0977430e+00	 1.6686728e-01	 1.1448130e+00	 1.5591725e-01	[ 1.0755949e+00]	 2.0517826e-01


.. parsed-literal::

      47	 1.1054209e+00	 1.6466158e-01	 1.1524820e+00	 1.5414421e-01	[ 1.0814157e+00]	 2.1638227e-01
      48	 1.1166671e+00	 1.6301933e-01	 1.1637917e+00	 1.5282195e-01	[ 1.0893705e+00]	 1.8398643e-01


.. parsed-literal::

      49	 1.1256602e+00	 1.6212979e-01	 1.1728873e+00	 1.5199939e-01	[ 1.0944660e+00]	 2.1364808e-01
      50	 1.1337834e+00	 1.6105250e-01	 1.1812942e+00	 1.5076359e-01	[ 1.0998276e+00]	 1.8458629e-01


.. parsed-literal::

      51	 1.1497790e+00	 1.5879210e-01	 1.1975779e+00	 1.4829800e-01	[ 1.1118301e+00]	 2.0617890e-01
      52	 1.1593015e+00	 1.5773811e-01	 1.2076178e+00	 1.4718968e-01	[ 1.1229016e+00]	 1.7094517e-01


.. parsed-literal::

      53	 1.1699160e+00	 1.5705347e-01	 1.2180481e+00	 1.4634182e-01	[ 1.1355390e+00]	 1.9988537e-01
      54	 1.1793401e+00	 1.5644915e-01	 1.2277144e+00	 1.4553585e-01	[ 1.1470710e+00]	 1.9925165e-01


.. parsed-literal::

      55	 1.1900272e+00	 1.5598766e-01	 1.2386386e+00	 1.4493828e-01	[ 1.1592439e+00]	 2.0736003e-01


.. parsed-literal::

      56	 1.1962111e+00	 1.5714256e-01	 1.2462385e+00	 1.4435150e-01	[ 1.1644723e+00]	 2.0646548e-01


.. parsed-literal::

      57	 1.2111113e+00	 1.5619691e-01	 1.2607337e+00	 1.4343938e-01	[ 1.1803231e+00]	 2.0144224e-01
      58	 1.2168892e+00	 1.5579058e-01	 1.2664525e+00	 1.4285808e-01	[ 1.1839695e+00]	 1.7046952e-01


.. parsed-literal::

      59	 1.2283983e+00	 1.5487828e-01	 1.2784783e+00	 1.4128514e-01	[ 1.1877787e+00]	 1.8265748e-01


.. parsed-literal::

      60	 1.2361565e+00	 1.5332913e-01	 1.2865170e+00	 1.3985559e-01	[ 1.1905051e+00]	 2.1786165e-01


.. parsed-literal::

      61	 1.2444586e+00	 1.5247765e-01	 1.2947884e+00	 1.3912997e-01	[ 1.1930423e+00]	 2.0720696e-01


.. parsed-literal::

      62	 1.2520906e+00	 1.5223281e-01	 1.3026650e+00	 1.3912739e-01	[ 1.1956132e+00]	 2.1979165e-01
      63	 1.2573965e+00	 1.5226322e-01	 1.3078553e+00	 1.3940818e-01	[ 1.1975459e+00]	 1.9754672e-01


.. parsed-literal::

      64	 1.2681922e+00	 1.5247977e-01	 1.3189129e+00	 1.3875979e-01	[ 1.2040603e+00]	 1.9378591e-01
      65	 1.2767809e+00	 1.5206704e-01	 1.3278134e+00	 1.3804853e-01	  1.2017349e+00 	 1.7996383e-01


.. parsed-literal::

      66	 1.2837828e+00	 1.5124339e-01	 1.3349927e+00	 1.3694617e-01	[ 1.2065750e+00]	 1.7604017e-01


.. parsed-literal::

      67	 1.2893044e+00	 1.5065048e-01	 1.3403413e+00	 1.3633155e-01	[ 1.2138771e+00]	 2.1102428e-01


.. parsed-literal::

      68	 1.2971166e+00	 1.4962719e-01	 1.3481582e+00	 1.3550754e-01	[ 1.2204464e+00]	 2.0695138e-01


.. parsed-literal::

      69	 1.3024929e+00	 1.4992415e-01	 1.3537733e+00	 1.3605116e-01	  1.2200732e+00 	 2.0161605e-01


.. parsed-literal::

      70	 1.3091086e+00	 1.4947213e-01	 1.3602940e+00	 1.3571722e-01	[ 1.2271308e+00]	 2.1648574e-01
      71	 1.3129470e+00	 1.4939614e-01	 1.3642456e+00	 1.3552112e-01	[ 1.2303661e+00]	 2.0744514e-01


.. parsed-literal::

      72	 1.3182931e+00	 1.4933089e-01	 1.3697370e+00	 1.3513067e-01	[ 1.2359321e+00]	 2.0647001e-01
      73	 1.3268005e+00	 1.4920213e-01	 1.3785460e+00	 1.3445125e-01	[ 1.2503875e+00]	 1.7401958e-01


.. parsed-literal::

      74	 1.3346281e+00	 1.4872058e-01	 1.3864577e+00	 1.3346466e-01	[ 1.2632309e+00]	 2.0996976e-01


.. parsed-literal::

      75	 1.3404090e+00	 1.4824463e-01	 1.3920938e+00	 1.3283773e-01	[ 1.2718630e+00]	 2.1033382e-01
      76	 1.3471017e+00	 1.4757930e-01	 1.3988118e+00	 1.3216958e-01	[ 1.2793843e+00]	 1.7738891e-01


.. parsed-literal::

      77	 1.3510524e+00	 1.4764432e-01	 1.4028355e+00	 1.3187434e-01	[ 1.2812739e+00]	 2.0684052e-01


.. parsed-literal::

      78	 1.3557192e+00	 1.4730825e-01	 1.4074843e+00	 1.3147799e-01	[ 1.2843769e+00]	 2.1632528e-01


.. parsed-literal::

      79	 1.3610544e+00	 1.4696164e-01	 1.4129315e+00	 1.3093294e-01	[ 1.2874312e+00]	 2.0584941e-01
      80	 1.3664372e+00	 1.4682836e-01	 1.4183881e+00	 1.3039683e-01	[ 1.2907812e+00]	 1.9398952e-01


.. parsed-literal::

      81	 1.3720075e+00	 1.4642093e-01	 1.4240720e+00	 1.2969499e-01	[ 1.2947304e+00]	 1.9960022e-01


.. parsed-literal::

      82	 1.3786394e+00	 1.4665715e-01	 1.4305815e+00	 1.2948631e-01	[ 1.3031453e+00]	 2.0592785e-01


.. parsed-literal::

      83	 1.3815236e+00	 1.4629999e-01	 1.4333827e+00	 1.2943154e-01	[ 1.3070135e+00]	 2.0387053e-01


.. parsed-literal::

      84	 1.3866954e+00	 1.4576559e-01	 1.4386133e+00	 1.2931293e-01	[ 1.3135129e+00]	 2.1058488e-01
      85	 1.3906859e+00	 1.4453468e-01	 1.4428497e+00	 1.2860916e-01	[ 1.3217075e+00]	 1.6731024e-01


.. parsed-literal::

      86	 1.3957518e+00	 1.4425696e-01	 1.4478323e+00	 1.2828805e-01	[ 1.3247557e+00]	 2.0630980e-01
      87	 1.3984881e+00	 1.4425140e-01	 1.4505592e+00	 1.2810849e-01	[ 1.3253072e+00]	 1.9644427e-01


.. parsed-literal::

      88	 1.4013581e+00	 1.4422624e-01	 1.4534635e+00	 1.2784840e-01	[ 1.3254327e+00]	 2.1402383e-01


.. parsed-literal::

      89	 1.4051770e+00	 1.4422784e-01	 1.4575031e+00	 1.2778511e-01	  1.3243419e+00 	 2.0824695e-01


.. parsed-literal::

      90	 1.4088073e+00	 1.4420261e-01	 1.4610804e+00	 1.2767621e-01	  1.3229927e+00 	 2.0195556e-01
      91	 1.4112951e+00	 1.4413286e-01	 1.4635811e+00	 1.2770029e-01	  1.3226121e+00 	 1.7384458e-01


.. parsed-literal::

      92	 1.4145154e+00	 1.4400815e-01	 1.4668522e+00	 1.2782860e-01	  1.3226987e+00 	 2.1479130e-01
      93	 1.4179466e+00	 1.4378895e-01	 1.4705334e+00	 1.2789661e-01	  1.3185579e+00 	 1.8173552e-01


.. parsed-literal::

      94	 1.4222343e+00	 1.4382494e-01	 1.4748034e+00	 1.2803208e-01	  1.3224709e+00 	 2.1086860e-01


.. parsed-literal::

      95	 1.4248530e+00	 1.4362211e-01	 1.4774036e+00	 1.2780324e-01	[ 1.3262475e+00]	 2.0265722e-01


.. parsed-literal::

      96	 1.4274979e+00	 1.4349340e-01	 1.4800460e+00	 1.2780753e-01	[ 1.3270988e+00]	 2.0378876e-01
      97	 1.4306585e+00	 1.4340230e-01	 1.4832815e+00	 1.2782425e-01	  1.3263312e+00 	 1.8128252e-01


.. parsed-literal::

      98	 1.4348372e+00	 1.4334011e-01	 1.4875147e+00	 1.2817285e-01	  1.3219291e+00 	 2.0757890e-01
      99	 1.4380983e+00	 1.4326232e-01	 1.4908254e+00	 1.2839016e-01	  1.3175483e+00 	 2.0014095e-01


.. parsed-literal::

     100	 1.4409664e+00	 1.4300684e-01	 1.4937108e+00	 1.2835834e-01	  1.3157122e+00 	 2.1484232e-01


.. parsed-literal::

     101	 1.4442204e+00	 1.4265754e-01	 1.4970037e+00	 1.2822499e-01	  1.3196774e+00 	 2.0887399e-01
     102	 1.4459771e+00	 1.4247130e-01	 1.4989665e+00	 1.2808238e-01	  1.3244855e+00 	 1.8480229e-01


.. parsed-literal::

     103	 1.4489915e+00	 1.4227155e-01	 1.5018890e+00	 1.2797467e-01	[ 1.3291554e+00]	 2.0950580e-01
     104	 1.4502792e+00	 1.4221284e-01	 1.5031820e+00	 1.2793762e-01	[ 1.3312426e+00]	 1.9543099e-01


.. parsed-literal::

     105	 1.4522326e+00	 1.4207895e-01	 1.5051947e+00	 1.2794491e-01	[ 1.3324060e+00]	 2.1369100e-01
     106	 1.4553746e+00	 1.4173028e-01	 1.5084206e+00	 1.2791563e-01	  1.3311437e+00 	 1.9721961e-01


.. parsed-literal::

     107	 1.4574128e+00	 1.4147313e-01	 1.5105583e+00	 1.2797271e-01	[ 1.3327843e+00]	 3.3010459e-01
     108	 1.4602256e+00	 1.4110077e-01	 1.5133897e+00	 1.2787337e-01	  1.3308037e+00 	 2.0480609e-01


.. parsed-literal::

     109	 1.4627929e+00	 1.4081867e-01	 1.5159528e+00	 1.2771858e-01	  1.3294768e+00 	 2.1404028e-01


.. parsed-literal::

     110	 1.4647103e+00	 1.4023149e-01	 1.5179557e+00	 1.2719194e-01	  1.3316485e+00 	 2.0570731e-01


.. parsed-literal::

     111	 1.4672698e+00	 1.4031645e-01	 1.5204392e+00	 1.2721702e-01	[ 1.3338931e+00]	 2.1816826e-01
     112	 1.4684755e+00	 1.4036073e-01	 1.5216537e+00	 1.2721163e-01	[ 1.3354663e+00]	 1.7937565e-01


.. parsed-literal::

     113	 1.4700098e+00	 1.4037179e-01	 1.5232537e+00	 1.2717277e-01	[ 1.3356414e+00]	 2.1294498e-01


.. parsed-literal::

     114	 1.4721001e+00	 1.4022320e-01	 1.5254233e+00	 1.2705589e-01	  1.3342422e+00 	 2.1318531e-01


.. parsed-literal::

     115	 1.4735659e+00	 1.3936740e-01	 1.5271808e+00	 1.2648786e-01	  1.3242827e+00 	 2.0894146e-01


.. parsed-literal::

     116	 1.4774548e+00	 1.3922411e-01	 1.5309590e+00	 1.2635316e-01	  1.3294116e+00 	 2.1822786e-01


.. parsed-literal::

     117	 1.4787850e+00	 1.3904213e-01	 1.5322182e+00	 1.2617194e-01	  1.3319614e+00 	 2.1254659e-01
     118	 1.4805933e+00	 1.3861952e-01	 1.5340226e+00	 1.2577405e-01	  1.3351483e+00 	 1.9247437e-01


.. parsed-literal::

     119	 1.4823049e+00	 1.3795303e-01	 1.5358081e+00	 1.2527740e-01	[ 1.3375993e+00]	 2.1735978e-01


.. parsed-literal::

     120	 1.4841623e+00	 1.3787169e-01	 1.5376449e+00	 1.2517656e-01	[ 1.3381264e+00]	 2.1789956e-01


.. parsed-literal::

     121	 1.4862155e+00	 1.3766632e-01	 1.5397675e+00	 1.2507379e-01	  1.3360277e+00 	 2.1183062e-01
     122	 1.4876335e+00	 1.3755425e-01	 1.5412327e+00	 1.2510083e-01	  1.3326073e+00 	 1.9991112e-01


.. parsed-literal::

     123	 1.4891267e+00	 1.3709291e-01	 1.5429641e+00	 1.2518249e-01	  1.3248789e+00 	 2.0376682e-01
     124	 1.4919916e+00	 1.3704600e-01	 1.5457476e+00	 1.2523761e-01	  1.3215902e+00 	 1.9960570e-01


.. parsed-literal::

     125	 1.4932928e+00	 1.3705333e-01	 1.5470042e+00	 1.2527794e-01	  1.3214541e+00 	 2.1563959e-01
     126	 1.4950151e+00	 1.3690544e-01	 1.5487215e+00	 1.2524372e-01	  1.3217731e+00 	 1.9647741e-01


.. parsed-literal::

     127	 1.4968327e+00	 1.3694541e-01	 1.5505441e+00	 1.2534552e-01	  1.3226162e+00 	 2.0810533e-01


.. parsed-literal::

     128	 1.4983617e+00	 1.3680940e-01	 1.5520543e+00	 1.2523835e-01	  1.3256634e+00 	 2.1794868e-01
     129	 1.5000941e+00	 1.3665235e-01	 1.5537875e+00	 1.2513476e-01	  1.3286740e+00 	 1.9562125e-01


.. parsed-literal::

     130	 1.5013405e+00	 1.3654324e-01	 1.5550529e+00	 1.2513563e-01	  1.3281098e+00 	 2.0484972e-01


.. parsed-literal::

     131	 1.5047741e+00	 1.3607680e-01	 1.5586018e+00	 1.2529345e-01	  1.3224807e+00 	 2.1540952e-01


.. parsed-literal::

     132	 1.5063187e+00	 1.3591083e-01	 1.5602215e+00	 1.2543803e-01	  1.3171823e+00 	 3.0780029e-01


.. parsed-literal::

     133	 1.5080955e+00	 1.3566366e-01	 1.5619975e+00	 1.2553436e-01	  1.3147767e+00 	 2.0549941e-01


.. parsed-literal::

     134	 1.5099437e+00	 1.3539199e-01	 1.5638512e+00	 1.2562179e-01	  1.3100943e+00 	 2.0203543e-01
     135	 1.5111732e+00	 1.3503909e-01	 1.5651200e+00	 1.2560279e-01	  1.3121217e+00 	 1.9424415e-01


.. parsed-literal::

     136	 1.5126356e+00	 1.3494072e-01	 1.5665507e+00	 1.2555846e-01	  1.3125404e+00 	 2.0877171e-01


.. parsed-literal::

     137	 1.5148179e+00	 1.3468200e-01	 1.5687271e+00	 1.2541251e-01	  1.3143536e+00 	 2.0913959e-01
     138	 1.5159119e+00	 1.3457064e-01	 1.5698471e+00	 1.2535128e-01	  1.3150212e+00 	 1.7464685e-01


.. parsed-literal::

     139	 1.5173449e+00	 1.3450390e-01	 1.5713688e+00	 1.2536262e-01	  1.3152906e+00 	 1.7593360e-01
     140	 1.5185017e+00	 1.3429865e-01	 1.5726080e+00	 1.2538448e-01	  1.3147585e+00 	 1.9754767e-01


.. parsed-literal::

     141	 1.5193755e+00	 1.3433774e-01	 1.5734602e+00	 1.2540887e-01	  1.3150240e+00 	 2.0329976e-01


.. parsed-literal::

     142	 1.5207147e+00	 1.3437160e-01	 1.5748155e+00	 1.2545475e-01	  1.3152472e+00 	 2.1426868e-01


.. parsed-literal::

     143	 1.5222388e+00	 1.3434014e-01	 1.5763746e+00	 1.2538281e-01	  1.3144358e+00 	 2.1172166e-01


.. parsed-literal::

     144	 1.5240927e+00	 1.3433817e-01	 1.5782653e+00	 1.2526115e-01	  1.3147594e+00 	 2.1437097e-01
     145	 1.5257486e+00	 1.3415178e-01	 1.5799026e+00	 1.2509727e-01	  1.3104893e+00 	 1.8591571e-01


.. parsed-literal::

     146	 1.5271437e+00	 1.3394732e-01	 1.5812256e+00	 1.2492460e-01	  1.3091002e+00 	 2.0407820e-01


.. parsed-literal::

     147	 1.5285871e+00	 1.3378941e-01	 1.5826238e+00	 1.2484890e-01	  1.3043721e+00 	 2.0888042e-01


.. parsed-literal::

     148	 1.5301767e+00	 1.3360055e-01	 1.5842319e+00	 1.2473529e-01	  1.2946521e+00 	 2.1834421e-01
     149	 1.5316732e+00	 1.3345489e-01	 1.5857540e+00	 1.2470071e-01	  1.2917092e+00 	 1.7840695e-01


.. parsed-literal::

     150	 1.5329113e+00	 1.3341427e-01	 1.5870377e+00	 1.2464611e-01	  1.2861794e+00 	 1.8153930e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.12 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8b081dbb80>



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

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.77 s, sys: 43.9 ms, total: 1.82 s
    Wall time: 571 ms


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

