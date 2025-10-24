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
       1	-3.4609412e-01	 3.2101226e-01	-3.3647941e-01	 3.1859650e-01	[-3.3222347e-01]	 5.6970406e-01


.. parsed-literal::

       2	-2.7336118e-01	 3.0994586e-01	-2.4861665e-01	 3.0875794e-01	[-2.4506447e-01]	 2.8489685e-01


.. parsed-literal::

       3	-2.2706456e-01	 2.8782199e-01	-1.8287920e-01	 2.9010922e-01	[-1.9259539e-01]	 3.2540464e-01
       4	-1.9588371e-01	 2.6456785e-01	-1.5654234e-01	 2.6947734e-01	[-1.8399074e-01]	 1.9767046e-01


.. parsed-literal::

       5	-9.5950938e-02	 2.5547118e-01	-6.1661983e-02	 2.5818722e-01	[-7.4276354e-02]	 2.1991420e-01


.. parsed-literal::

       6	-6.7121609e-02	 2.5129840e-01	-3.7467481e-02	 2.5617964e-01	[-5.5760692e-02]	 2.0986819e-01


.. parsed-literal::

       7	-4.7537004e-02	 2.4789863e-01	-2.4728240e-02	 2.5240889e-01	[-4.1829616e-02]	 2.1953106e-01


.. parsed-literal::

       8	-3.7871774e-02	 2.4626544e-01	-1.8062474e-02	 2.5063583e-01	[-3.4520337e-02]	 2.3087811e-01


.. parsed-literal::

       9	-2.5139912e-02	 2.4382646e-01	-7.7649638e-03	 2.4860352e-01	[-2.5172628e-02]	 2.2088575e-01


.. parsed-literal::

      10	-1.3519669e-02	 2.4159380e-01	 1.9456721e-03	 2.4713780e-01	[-1.7179736e-02]	 2.1089578e-01


.. parsed-literal::

      11	-1.0251487e-02	 2.4130990e-01	 3.9741083e-03	 2.4556459e-01	[-1.0026057e-02]	 2.2499108e-01


.. parsed-literal::

      12	-6.9641127e-03	 2.4054850e-01	 7.2403420e-03	 2.4463581e-01	[-5.9130084e-03]	 2.2840214e-01


.. parsed-literal::

      13	-3.6009604e-03	 2.3984827e-01	 1.0978151e-02	 2.4370328e-01	[-1.8493865e-03]	 2.1824574e-01


.. parsed-literal::

      14	 2.8663446e-02	 2.3465973e-01	 4.5335891e-02	 2.3878993e-01	[ 3.0653976e-02]	 3.5863829e-01


.. parsed-literal::

      15	 6.4012678e-02	 2.2537123e-01	 8.6358935e-02	 2.2523307e-01	[ 8.4552121e-02]	 2.2656012e-01


.. parsed-literal::

      16	 1.5041313e-01	 2.2302363e-01	 1.7633184e-01	 2.2864469e-01	[ 1.6584427e-01]	 2.2188401e-01


.. parsed-literal::

      17	 2.0403001e-01	 2.1856004e-01	 2.3372584e-01	 2.2555469e-01	[ 2.1581967e-01]	 2.2306657e-01


.. parsed-literal::

      18	 2.9483840e-01	 2.1431634e-01	 3.2518720e-01	 2.2067167e-01	[ 3.0013079e-01]	 2.2312737e-01


.. parsed-literal::

      19	 3.4716348e-01	 2.0805478e-01	 3.7851304e-01	 2.1664720e-01	[ 3.4624378e-01]	 2.0715642e-01


.. parsed-literal::

      20	 4.7305114e-01	 1.9994256e-01	 5.1016591e-01	 2.0865924e-01	[ 4.4230134e-01]	 2.1234822e-01


.. parsed-literal::

      21	 5.2571157e-01	 2.0188020e-01	 5.6228279e-01	 2.0878587e-01	[ 5.1523888e-01]	 2.2814512e-01


.. parsed-literal::

      22	 5.8520676e-01	 1.9318710e-01	 6.2415051e-01	 2.0108787e-01	[ 5.5857750e-01]	 2.0374560e-01


.. parsed-literal::

      23	 6.2764011e-01	 1.8664178e-01	 6.6750157e-01	 1.9721142e-01	[ 5.8867613e-01]	 2.0707607e-01


.. parsed-literal::

      24	 6.6995094e-01	 1.8380913e-01	 7.0925160e-01	 1.9436396e-01	[ 6.3545188e-01]	 2.1947551e-01
      25	 7.2009016e-01	 1.8181101e-01	 7.5903634e-01	 1.9095662e-01	[ 6.9254321e-01]	 2.0583916e-01


.. parsed-literal::

      26	 7.7046430e-01	 1.8280388e-01	 8.1004010e-01	 1.8933827e-01	[ 7.5148941e-01]	 2.1979427e-01


.. parsed-literal::

      27	 8.1066056e-01	 1.8785554e-01	 8.5162408e-01	 1.9280187e-01	[ 7.9756850e-01]	 2.1561718e-01


.. parsed-literal::

      28	 8.2790405e-01	 1.9175816e-01	 8.6826477e-01	 1.9531195e-01	[ 8.1489830e-01]	 2.2074914e-01


.. parsed-literal::

      29	 8.5665658e-01	 1.9026793e-01	 8.9731268e-01	 1.9282911e-01	[ 8.4027577e-01]	 2.1672630e-01


.. parsed-literal::

      30	 8.7581886e-01	 1.8642290e-01	 9.1829798e-01	 1.8791344e-01	[ 8.5411152e-01]	 2.2566509e-01


.. parsed-literal::

      31	 8.9478461e-01	 1.8457806e-01	 9.3658896e-01	 1.8677925e-01	[ 8.7462668e-01]	 2.0104194e-01


.. parsed-literal::

      32	 9.1959522e-01	 1.8243217e-01	 9.6128685e-01	 1.8430437e-01	[ 9.0354443e-01]	 2.2081304e-01


.. parsed-literal::

      33	 9.4329567e-01	 1.7831760e-01	 9.8537456e-01	 1.8198210e-01	[ 9.2677796e-01]	 2.1744990e-01


.. parsed-literal::

      34	 9.6644471e-01	 1.7311299e-01	 1.0089533e+00	 1.7748677e-01	[ 9.4560471e-01]	 2.2003794e-01


.. parsed-literal::

      35	 9.8675627e-01	 1.6992845e-01	 1.0297459e+00	 1.7554820e-01	[ 9.6074547e-01]	 2.2881007e-01


.. parsed-literal::

      36	 1.0116233e+00	 1.6635039e-01	 1.0559154e+00	 1.7242655e-01	[ 9.7633971e-01]	 2.3623300e-01


.. parsed-literal::

      37	 1.0316505e+00	 1.6398153e-01	 1.0768490e+00	 1.6905492e-01	[ 9.9587856e-01]	 2.2800374e-01


.. parsed-literal::

      38	 1.0530782e+00	 1.6184832e-01	 1.0990237e+00	 1.6573562e-01	[ 1.0085121e+00]	 2.2219610e-01


.. parsed-literal::

      39	 1.0700494e+00	 1.5990784e-01	 1.1164464e+00	 1.6251279e-01	[ 1.0189542e+00]	 2.2855878e-01


.. parsed-literal::

      40	 1.0897494e+00	 1.5741783e-01	 1.1362416e+00	 1.5962565e-01	[ 1.0315031e+00]	 2.0573306e-01


.. parsed-literal::

      41	 1.1105864e+00	 1.5443102e-01	 1.1574021e+00	 1.5671214e-01	[ 1.0440919e+00]	 2.0458651e-01
      42	 1.1292765e+00	 1.5188876e-01	 1.1761685e+00	 1.5524003e-01	  1.0423246e+00 	 1.9572282e-01


.. parsed-literal::

      43	 1.1451754e+00	 1.4909234e-01	 1.1917325e+00	 1.5286779e-01	[ 1.0574991e+00]	 2.1307158e-01


.. parsed-literal::

      44	 1.1574521e+00	 1.4783437e-01	 1.2042626e+00	 1.5245782e-01	[ 1.0605909e+00]	 2.1522260e-01
      45	 1.1663343e+00	 1.4480618e-01	 1.2137379e+00	 1.4915147e-01	  1.0493845e+00 	 1.8963599e-01


.. parsed-literal::

      46	 1.1754093e+00	 1.4470720e-01	 1.2225661e+00	 1.4894154e-01	[ 1.0618638e+00]	 2.2985196e-01


.. parsed-literal::

      47	 1.1828763e+00	 1.4446051e-01	 1.2299514e+00	 1.4879303e-01	[ 1.0682608e+00]	 2.1857047e-01


.. parsed-literal::

      48	 1.1974834e+00	 1.4360034e-01	 1.2448833e+00	 1.4807092e-01	[ 1.0699882e+00]	 2.1430707e-01


.. parsed-literal::

      49	 1.2063732e+00	 1.4454618e-01	 1.2543081e+00	 1.4900447e-01	[ 1.0739188e+00]	 2.1898723e-01


.. parsed-literal::

      50	 1.2167745e+00	 1.4328709e-01	 1.2647379e+00	 1.4765631e-01	  1.0719858e+00 	 2.1796751e-01


.. parsed-literal::

      51	 1.2258243e+00	 1.4205750e-01	 1.2739949e+00	 1.4622474e-01	  1.0667489e+00 	 2.0858979e-01


.. parsed-literal::

      52	 1.2352691e+00	 1.4125992e-01	 1.2837856e+00	 1.4495795e-01	  1.0714859e+00 	 2.2298741e-01


.. parsed-literal::

      53	 1.2495268e+00	 1.4051840e-01	 1.2983698e+00	 1.4374685e-01	[ 1.0807259e+00]	 2.3155713e-01


.. parsed-literal::

      54	 1.2613992e+00	 1.4023642e-01	 1.3107073e+00	 1.4326036e-01	[ 1.0909322e+00]	 2.2074747e-01


.. parsed-literal::

      55	 1.2716038e+00	 1.4075429e-01	 1.3209489e+00	 1.4410439e-01	[ 1.1200253e+00]	 2.2007155e-01


.. parsed-literal::

      56	 1.2796854e+00	 1.4089624e-01	 1.3289885e+00	 1.4415739e-01	[ 1.1304826e+00]	 2.1783352e-01


.. parsed-literal::

      57	 1.2881858e+00	 1.4026034e-01	 1.3375555e+00	 1.4335893e-01	[ 1.1447236e+00]	 2.1623397e-01


.. parsed-literal::

      58	 1.2962613e+00	 1.4026143e-01	 1.3458617e+00	 1.4361739e-01	[ 1.1544002e+00]	 2.1523952e-01


.. parsed-literal::

      59	 1.3060720e+00	 1.3819688e-01	 1.3557835e+00	 1.4128878e-01	[ 1.1726901e+00]	 2.2409630e-01


.. parsed-literal::

      60	 1.3151380e+00	 1.3831748e-01	 1.3647550e+00	 1.4144127e-01	[ 1.1852775e+00]	 2.2214794e-01


.. parsed-literal::

      61	 1.3249765e+00	 1.3800634e-01	 1.3746185e+00	 1.4090330e-01	[ 1.1992087e+00]	 2.2960758e-01


.. parsed-literal::

      62	 1.3345732e+00	 1.3869163e-01	 1.3845208e+00	 1.4167826e-01	[ 1.2102632e+00]	 2.2879767e-01


.. parsed-literal::

      63	 1.3373644e+00	 1.3949185e-01	 1.3878603e+00	 1.4120852e-01	[ 1.2188556e+00]	 2.1803284e-01


.. parsed-literal::

      64	 1.3464157e+00	 1.3886275e-01	 1.3965470e+00	 1.4123748e-01	[ 1.2234910e+00]	 2.3178887e-01


.. parsed-literal::

      65	 1.3509401e+00	 1.3930479e-01	 1.4012357e+00	 1.4205828e-01	  1.2227809e+00 	 2.1608257e-01


.. parsed-literal::

      66	 1.3564764e+00	 1.3973722e-01	 1.4071396e+00	 1.4274391e-01	  1.2192765e+00 	 2.2750020e-01


.. parsed-literal::

      67	 1.3627148e+00	 1.3992085e-01	 1.4138078e+00	 1.4312858e-01	  1.2125631e+00 	 2.1965289e-01


.. parsed-literal::

      68	 1.3688703e+00	 1.4027438e-01	 1.4200759e+00	 1.4341905e-01	  1.2232125e+00 	 2.3189306e-01


.. parsed-literal::

      69	 1.3723753e+00	 1.3983652e-01	 1.4234474e+00	 1.4282424e-01	[ 1.2269238e+00]	 2.2710323e-01


.. parsed-literal::

      70	 1.3796341e+00	 1.3946714e-01	 1.4309094e+00	 1.4236579e-01	[ 1.2293859e+00]	 2.1693397e-01


.. parsed-literal::

      71	 1.3864350e+00	 1.3909268e-01	 1.4376665e+00	 1.4244808e-01	  1.2243068e+00 	 2.3021054e-01


.. parsed-literal::

      72	 1.3916406e+00	 1.3903106e-01	 1.4429336e+00	 1.4265563e-01	[ 1.2308325e+00]	 2.2050762e-01


.. parsed-literal::

      73	 1.3954395e+00	 1.3890343e-01	 1.4467329e+00	 1.4267374e-01	  1.2307685e+00 	 2.2144771e-01


.. parsed-literal::

      74	 1.4006467e+00	 1.3890393e-01	 1.4520919e+00	 1.4311090e-01	  1.2243861e+00 	 2.1349454e-01


.. parsed-literal::

      75	 1.4054844e+00	 1.3858567e-01	 1.4569843e+00	 1.4289466e-01	  1.2265203e+00 	 2.1430516e-01


.. parsed-literal::

      76	 1.4121487e+00	 1.3830416e-01	 1.4638592e+00	 1.4312664e-01	  1.2227827e+00 	 2.3301840e-01


.. parsed-literal::

      77	 1.4167422e+00	 1.3754342e-01	 1.4684265e+00	 1.4227208e-01	  1.2238867e+00 	 2.3165417e-01


.. parsed-literal::

      78	 1.4207330e+00	 1.3734252e-01	 1.4724069e+00	 1.4213124e-01	[ 1.2329115e+00]	 2.1756387e-01


.. parsed-literal::

      79	 1.4243526e+00	 1.3704169e-01	 1.4760616e+00	 1.4188738e-01	[ 1.2380969e+00]	 2.2396922e-01


.. parsed-literal::

      80	 1.4285516e+00	 1.3704520e-01	 1.4802487e+00	 1.4204288e-01	[ 1.2440060e+00]	 2.2536993e-01


.. parsed-literal::

      81	 1.4324559e+00	 1.3688494e-01	 1.4841886e+00	 1.4204405e-01	[ 1.2491123e+00]	 2.0995927e-01
      82	 1.4358682e+00	 1.3626453e-01	 1.4878254e+00	 1.4148217e-01	[ 1.2553065e+00]	 1.9230461e-01


.. parsed-literal::

      83	 1.4393462e+00	 1.3614799e-01	 1.4912516e+00	 1.4145120e-01	[ 1.2590727e+00]	 2.2326446e-01


.. parsed-literal::

      84	 1.4418149e+00	 1.3606943e-01	 1.4937183e+00	 1.4150645e-01	[ 1.2604580e+00]	 2.0535493e-01


.. parsed-literal::

      85	 1.4445886e+00	 1.3613545e-01	 1.4964996e+00	 1.4176129e-01	  1.2600895e+00 	 2.0747185e-01


.. parsed-literal::

      86	 1.4490654e+00	 1.3593192e-01	 1.5009593e+00	 1.4167150e-01	[ 1.2608061e+00]	 2.2548819e-01


.. parsed-literal::

      87	 1.4533553e+00	 1.3569421e-01	 1.5053123e+00	 1.4134747e-01	  1.2596260e+00 	 2.1406865e-01


.. parsed-literal::

      88	 1.4568863e+00	 1.3536553e-01	 1.5088599e+00	 1.4098581e-01	[ 1.2615154e+00]	 2.1841073e-01


.. parsed-literal::

      89	 1.4604537e+00	 1.3495299e-01	 1.5125904e+00	 1.4053427e-01	[ 1.2658387e+00]	 2.3041558e-01
      90	 1.4635818e+00	 1.3473119e-01	 1.5159031e+00	 1.4023188e-01	  1.2652951e+00 	 2.0016170e-01


.. parsed-literal::

      91	 1.4669330e+00	 1.3476218e-01	 1.5194901e+00	 1.4060925e-01	[ 1.2706033e+00]	 2.1759486e-01


.. parsed-literal::

      92	 1.4690494e+00	 1.3477696e-01	 1.5215701e+00	 1.4077534e-01	[ 1.2709631e+00]	 2.1968794e-01


.. parsed-literal::

      93	 1.4722277e+00	 1.3479498e-01	 1.5247501e+00	 1.4114897e-01	  1.2684507e+00 	 2.0881176e-01
      94	 1.4752654e+00	 1.3481953e-01	 1.5278384e+00	 1.4165709e-01	  1.2648000e+00 	 1.8278575e-01


.. parsed-literal::

      95	 1.4786676e+00	 1.3466916e-01	 1.5312023e+00	 1.4165514e-01	  1.2588495e+00 	 2.0345640e-01


.. parsed-literal::

      96	 1.4813805e+00	 1.3448468e-01	 1.5339337e+00	 1.4151539e-01	  1.2540121e+00 	 2.2057223e-01


.. parsed-literal::

      97	 1.4835027e+00	 1.3442416e-01	 1.5361010e+00	 1.4148327e-01	  1.2457505e+00 	 2.1208572e-01


.. parsed-literal::

      98	 1.4859656e+00	 1.3415045e-01	 1.5386190e+00	 1.4132202e-01	  1.2440207e+00 	 2.1289992e-01


.. parsed-literal::

      99	 1.4886175e+00	 1.3412446e-01	 1.5413031e+00	 1.4153225e-01	  1.2427802e+00 	 2.2253990e-01


.. parsed-literal::

     100	 1.4920191e+00	 1.3409396e-01	 1.5447986e+00	 1.4218028e-01	  1.2464204e+00 	 2.1495581e-01


.. parsed-literal::

     101	 1.4945174e+00	 1.3398900e-01	 1.5472051e+00	 1.4210871e-01	  1.2454715e+00 	 2.1988201e-01


.. parsed-literal::

     102	 1.4961068e+00	 1.3391276e-01	 1.5487408e+00	 1.4201190e-01	  1.2493274e+00 	 2.2008681e-01


.. parsed-literal::

     103	 1.4989485e+00	 1.3381327e-01	 1.5516205e+00	 1.4220347e-01	  1.2527381e+00 	 2.1461749e-01


.. parsed-literal::

     104	 1.5005460e+00	 1.3368264e-01	 1.5533451e+00	 1.4218646e-01	  1.2467525e+00 	 2.1724200e-01


.. parsed-literal::

     105	 1.5025972e+00	 1.3357494e-01	 1.5554608e+00	 1.4230350e-01	  1.2458204e+00 	 2.2730136e-01


.. parsed-literal::

     106	 1.5052655e+00	 1.3339108e-01	 1.5582806e+00	 1.4247018e-01	  1.2402262e+00 	 2.2353387e-01


.. parsed-literal::

     107	 1.5070391e+00	 1.3323748e-01	 1.5601074e+00	 1.4240590e-01	  1.2368001e+00 	 2.3076773e-01


.. parsed-literal::

     108	 1.5099214e+00	 1.3303244e-01	 1.5631557e+00	 1.4293213e-01	  1.2209861e+00 	 2.2676611e-01


.. parsed-literal::

     109	 1.5125427e+00	 1.3294381e-01	 1.5657279e+00	 1.4292741e-01	  1.2170469e+00 	 2.0536900e-01


.. parsed-literal::

     110	 1.5144630e+00	 1.3298229e-01	 1.5675307e+00	 1.4298529e-01	  1.2191390e+00 	 2.2707438e-01


.. parsed-literal::

     111	 1.5175966e+00	 1.3310786e-01	 1.5705515e+00	 1.4354276e-01	  1.2114727e+00 	 2.2578192e-01


.. parsed-literal::

     112	 1.5194974e+00	 1.3338028e-01	 1.5725024e+00	 1.4414459e-01	  1.1925676e+00 	 2.2632718e-01


.. parsed-literal::

     113	 1.5216403e+00	 1.3332633e-01	 1.5746528e+00	 1.4427621e-01	  1.1833812e+00 	 2.2965717e-01


.. parsed-literal::

     114	 1.5240012e+00	 1.3335586e-01	 1.5770752e+00	 1.4439853e-01	  1.1658094e+00 	 2.1968079e-01


.. parsed-literal::

     115	 1.5255097e+00	 1.3324557e-01	 1.5786651e+00	 1.4438297e-01	  1.1586078e+00 	 2.1991944e-01
     116	 1.5276567e+00	 1.3319003e-01	 1.5808628e+00	 1.4428549e-01	  1.1512953e+00 	 1.9468832e-01


.. parsed-literal::

     117	 1.5298557e+00	 1.3305315e-01	 1.5831917e+00	 1.4409913e-01	  1.1353315e+00 	 2.1404266e-01
     118	 1.5317743e+00	 1.3280708e-01	 1.5851013e+00	 1.4385444e-01	  1.1430346e+00 	 1.9735909e-01


.. parsed-literal::

     119	 1.5331332e+00	 1.3269280e-01	 1.5864446e+00	 1.4382436e-01	  1.1423637e+00 	 2.0520926e-01


.. parsed-literal::

     120	 1.5362028e+00	 1.3230792e-01	 1.5896641e+00	 1.4364648e-01	  1.1374991e+00 	 2.2387576e-01
     121	 1.5373520e+00	 1.3205463e-01	 1.5909436e+00	 1.4331648e-01	  1.1325176e+00 	 2.0286322e-01


.. parsed-literal::

     122	 1.5390283e+00	 1.3206311e-01	 1.5925404e+00	 1.4326325e-01	  1.1370813e+00 	 2.2846293e-01


.. parsed-literal::

     123	 1.5401490e+00	 1.3195041e-01	 1.5936741e+00	 1.4306817e-01	  1.1401047e+00 	 2.2063637e-01


.. parsed-literal::

     124	 1.5416678e+00	 1.3172001e-01	 1.5952069e+00	 1.4268281e-01	  1.1470217e+00 	 2.2030568e-01


.. parsed-literal::

     125	 1.5439931e+00	 1.3138161e-01	 1.5975052e+00	 1.4224399e-01	  1.1615799e+00 	 2.1408463e-01


.. parsed-literal::

     126	 1.5455217e+00	 1.3116007e-01	 1.5990131e+00	 1.4179768e-01	  1.1667153e+00 	 3.5837650e-01


.. parsed-literal::

     127	 1.5470879e+00	 1.3099539e-01	 1.6005213e+00	 1.4164150e-01	  1.1755243e+00 	 2.2387600e-01


.. parsed-literal::

     128	 1.5483538e+00	 1.3092777e-01	 1.6017626e+00	 1.4160333e-01	  1.1755898e+00 	 2.2131205e-01


.. parsed-literal::

     129	 1.5497707e+00	 1.3085929e-01	 1.6032065e+00	 1.4156037e-01	  1.1738004e+00 	 2.1900034e-01


.. parsed-literal::

     130	 1.5515357e+00	 1.3089891e-01	 1.6050518e+00	 1.4153527e-01	  1.1620787e+00 	 2.2013521e-01


.. parsed-literal::

     131	 1.5532791e+00	 1.3088669e-01	 1.6068650e+00	 1.4149883e-01	  1.1519375e+00 	 2.1636271e-01


.. parsed-literal::

     132	 1.5549533e+00	 1.3088932e-01	 1.6086615e+00	 1.4149471e-01	  1.1414118e+00 	 2.2566795e-01


.. parsed-literal::

     133	 1.5561756e+00	 1.3088036e-01	 1.6099256e+00	 1.4137934e-01	  1.1274870e+00 	 2.2736669e-01


.. parsed-literal::

     134	 1.5574275e+00	 1.3078572e-01	 1.6111735e+00	 1.4122969e-01	  1.1231530e+00 	 2.1833277e-01


.. parsed-literal::

     135	 1.5606475e+00	 1.3052236e-01	 1.6144759e+00	 1.4079613e-01	  1.0984823e+00 	 2.2387433e-01


.. parsed-literal::

     136	 1.5618366e+00	 1.3041144e-01	 1.6157651e+00	 1.4053280e-01	  1.0777799e+00 	 3.6753249e-01


.. parsed-literal::

     137	 1.5633222e+00	 1.3036045e-01	 1.6172732e+00	 1.4039736e-01	  1.0666405e+00 	 2.1869779e-01


.. parsed-literal::

     138	 1.5654834e+00	 1.3036952e-01	 1.6195237e+00	 1.4025416e-01	  1.0432939e+00 	 2.1061993e-01


.. parsed-literal::

     139	 1.5664632e+00	 1.3030577e-01	 1.6205707e+00	 1.4007924e-01	  1.0371364e+00 	 2.1631551e-01


.. parsed-literal::

     140	 1.5677692e+00	 1.3026191e-01	 1.6218568e+00	 1.3989576e-01	  1.0368018e+00 	 2.2552490e-01


.. parsed-literal::

     141	 1.5696127e+00	 1.3004717e-01	 1.6236853e+00	 1.3946274e-01	  1.0397267e+00 	 2.1096730e-01


.. parsed-literal::

     142	 1.5710405e+00	 1.2979052e-01	 1.6250944e+00	 1.3885691e-01	  1.0401919e+00 	 2.2677851e-01


.. parsed-literal::

     143	 1.5725138e+00	 1.2934374e-01	 1.6265490e+00	 1.3800485e-01	  1.0439429e+00 	 2.3105526e-01


.. parsed-literal::

     144	 1.5738033e+00	 1.2924918e-01	 1.6278165e+00	 1.3778601e-01	  1.0463364e+00 	 2.0557499e-01


.. parsed-literal::

     145	 1.5748123e+00	 1.2922870e-01	 1.6288409e+00	 1.3768362e-01	  1.0403917e+00 	 2.0796108e-01


.. parsed-literal::

     146	 1.5760138e+00	 1.2919830e-01	 1.6301004e+00	 1.3756119e-01	  1.0368292e+00 	 2.2153354e-01


.. parsed-literal::

     147	 1.5775276e+00	 1.2910261e-01	 1.6317368e+00	 1.3729068e-01	  1.0323527e+00 	 2.1051598e-01


.. parsed-literal::

     148	 1.5788413e+00	 1.2915373e-01	 1.6330702e+00	 1.3731665e-01	  1.0326346e+00 	 2.1752906e-01


.. parsed-literal::

     149	 1.5799060e+00	 1.2920737e-01	 1.6341546e+00	 1.3735997e-01	  1.0362624e+00 	 2.0610476e-01


.. parsed-literal::

     150	 1.5811159e+00	 1.2924434e-01	 1.6353966e+00	 1.3734188e-01	  1.0336567e+00 	 2.1505284e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 14s, sys: 1.2 s, total: 2min 15s
    Wall time: 33.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f356c8e7dc0>



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
    CPU times: user 1.71 s, sys: 52.9 ms, total: 1.76 s
    Wall time: 563 ms


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

