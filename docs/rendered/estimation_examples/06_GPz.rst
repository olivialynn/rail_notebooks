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
       1	-3.3949498e-01	 3.1944117e-01	-3.2987871e-01	 3.2556068e-01	[-3.4177658e-01]	 4.5856643e-01


.. parsed-literal::

       2	-2.7069063e-01	 3.0964430e-01	-2.4744100e-01	 3.1392732e-01	[-2.6085757e-01]	 2.3069215e-01


.. parsed-literal::

       3	-2.2296640e-01	 2.8673846e-01	-1.7888882e-01	 2.9035490e-01	[-1.9344517e-01]	 2.8701282e-01


.. parsed-literal::

       4	-1.9031093e-01	 2.6394384e-01	-1.5003780e-01	 2.7072981e-01	[-1.8152465e-01]	 2.1564031e-01


.. parsed-literal::

       5	-9.2741613e-02	 2.5541119e-01	-5.8988679e-02	 2.6062853e-01	[-8.0641444e-02]	 2.1032953e-01
       6	-6.6088220e-02	 2.5116035e-01	-3.7141925e-02	 2.5855482e-01	[-5.6402442e-02]	 1.8321609e-01


.. parsed-literal::

       7	-4.7242973e-02	 2.4790983e-01	-2.4645693e-02	 2.5211411e-01	[-3.9433625e-02]	 2.1404958e-01


.. parsed-literal::

       8	-3.6800121e-02	 2.4617146e-01	-1.7369733e-02	 2.4981423e-01	[-3.1356029e-02]	 2.1978593e-01
       9	-2.4096340e-02	 2.4378319e-01	-7.1940402e-03	 2.4722324e-01	[-2.0433582e-02]	 1.8910837e-01


.. parsed-literal::

      10	-1.5674077e-02	 2.4206586e-01	-7.3029790e-04	 2.4504896e-01	[-1.2228915e-02]	 2.0945787e-01
      11	-8.6427674e-03	 2.4095026e-01	 5.4537840e-03	 2.4407754e-01	[-3.9795576e-03]	 2.0006943e-01


.. parsed-literal::

      12	-6.5913828e-03	 2.4054867e-01	 7.4035667e-03	 2.4366156e-01	[-1.8411435e-03]	 1.7343211e-01


.. parsed-literal::

      13	 1.0000913e-03	 2.3904526e-01	 1.5225746e-02	 2.4178690e-01	[ 7.9677209e-03]	 2.1044254e-01


.. parsed-literal::

      14	 9.6716255e-02	 2.2706888e-01	 1.1727267e-01	 2.2946371e-01	[ 1.1563078e-01]	 3.0946088e-01


.. parsed-literal::

      15	 1.8166782e-01	 2.2562449e-01	 2.0910481e-01	 2.2644623e-01	[ 2.1795526e-01]	 3.2241535e-01


.. parsed-literal::

      16	 2.7413420e-01	 2.1590360e-01	 3.0417967e-01	 2.1472407e-01	[ 3.1959978e-01]	 2.1072292e-01
      17	 3.4133953e-01	 2.0939548e-01	 3.7396948e-01	 2.0791615e-01	[ 3.9274802e-01]	 1.7601752e-01


.. parsed-literal::

      18	 3.9275153e-01	 2.0627659e-01	 4.2616579e-01	 2.0266348e-01	[ 4.4893957e-01]	 2.1026015e-01
      19	 4.4381031e-01	 2.0172714e-01	 4.7763524e-01	 1.9894031e-01	[ 4.9494188e-01]	 1.8663073e-01


.. parsed-literal::

      20	 5.7697661e-01	 1.9566474e-01	 6.1412720e-01	 1.9573872e-01	[ 6.0174313e-01]	 2.0369434e-01


.. parsed-literal::

      21	 6.3896137e-01	 1.9254598e-01	 6.7815055e-01	 1.9184408e-01	[ 6.6019192e-01]	 2.1206093e-01


.. parsed-literal::

      22	 6.7876342e-01	 1.8993677e-01	 7.1780772e-01	 1.8778820e-01	[ 6.9471077e-01]	 2.1153784e-01


.. parsed-literal::

      23	 7.0913622e-01	 1.8659083e-01	 7.4669921e-01	 1.8367711e-01	[ 7.3251866e-01]	 2.1496701e-01
      24	 7.4853492e-01	 1.8827462e-01	 7.8668027e-01	 1.8637818e-01	[ 7.7856683e-01]	 1.7084646e-01


.. parsed-literal::

      25	 8.0025313e-01	 1.8388520e-01	 8.3837378e-01	 1.8121089e-01	[ 8.3389292e-01]	 2.0467925e-01


.. parsed-literal::

      26	 8.3698300e-01	 1.8114142e-01	 8.7616043e-01	 1.7801992e-01	[ 8.6505816e-01]	 2.0539308e-01


.. parsed-literal::

      27	 8.6091310e-01	 1.7567376e-01	 9.0153609e-01	 1.7032537e-01	[ 8.8914821e-01]	 2.1029782e-01


.. parsed-literal::

      28	 8.8188669e-01	 1.7372196e-01	 9.2306888e-01	 1.6802643e-01	[ 9.1384378e-01]	 2.1165085e-01
      29	 9.0555749e-01	 1.7105001e-01	 9.4734752e-01	 1.6827159e-01	[ 9.3851607e-01]	 1.9856524e-01


.. parsed-literal::

      30	 9.3070109e-01	 1.6838001e-01	 9.7298547e-01	 1.6611328e-01	[ 9.6806666e-01]	 2.0135164e-01


.. parsed-literal::

      31	 9.5497463e-01	 1.6667186e-01	 9.9771600e-01	 1.6552535e-01	[ 9.9995995e-01]	 2.0981050e-01


.. parsed-literal::

      32	 9.7485813e-01	 1.6524290e-01	 1.0182006e+00	 1.6774375e-01	[ 1.0100086e+00]	 2.0517921e-01


.. parsed-literal::

      33	 9.8866443e-01	 1.6397775e-01	 1.0323728e+00	 1.6623215e-01	[ 1.0200495e+00]	 2.1243572e-01


.. parsed-literal::

      34	 1.0133636e+00	 1.6263212e-01	 1.0581760e+00	 1.6469140e-01	[ 1.0389212e+00]	 2.0232582e-01
      35	 1.0289748e+00	 1.6032298e-01	 1.0749653e+00	 1.6255915e-01	[ 1.0500904e+00]	 2.0015907e-01


.. parsed-literal::

      36	 1.0426697e+00	 1.5995907e-01	 1.0881619e+00	 1.6212992e-01	[ 1.0694018e+00]	 2.0650196e-01


.. parsed-literal::

      37	 1.0507096e+00	 1.5964765e-01	 1.0962252e+00	 1.6134073e-01	[ 1.0783973e+00]	 2.1620584e-01


.. parsed-literal::

      38	 1.0663859e+00	 1.5867877e-01	 1.1123001e+00	 1.5979230e-01	[ 1.0917209e+00]	 2.1034265e-01
      39	 1.0792875e+00	 1.5631581e-01	 1.1256271e+00	 1.5753853e-01	[ 1.1023980e+00]	 1.7683196e-01


.. parsed-literal::

      40	 1.0914910e+00	 1.5498292e-01	 1.1377061e+00	 1.5715352e-01	[ 1.1118513e+00]	 1.7225981e-01
      41	 1.1036225e+00	 1.5306014e-01	 1.1501771e+00	 1.5588897e-01	[ 1.1209449e+00]	 1.8388939e-01


.. parsed-literal::

      42	 1.1135359e+00	 1.5147923e-01	 1.1607251e+00	 1.5486763e-01	[ 1.1277469e+00]	 2.1654367e-01
      43	 1.1228310e+00	 1.5056184e-01	 1.1702596e+00	 1.5479565e-01	[ 1.1314641e+00]	 1.7887521e-01


.. parsed-literal::

      44	 1.1304693e+00	 1.5021152e-01	 1.1778614e+00	 1.5401337e-01	[ 1.1358643e+00]	 2.1431828e-01
      45	 1.1434780e+00	 1.4977603e-01	 1.1909454e+00	 1.5315801e-01	[ 1.1436160e+00]	 1.8601227e-01


.. parsed-literal::

      46	 1.1530238e+00	 1.4946867e-01	 1.2002739e+00	 1.5263440e-01	[ 1.1498802e+00]	 2.0633650e-01
      47	 1.1628546e+00	 1.4865980e-01	 1.2101877e+00	 1.5310168e-01	[ 1.1582969e+00]	 1.7741895e-01


.. parsed-literal::

      48	 1.1762957e+00	 1.4766798e-01	 1.2240010e+00	 1.5403727e-01	[ 1.1681590e+00]	 1.9530416e-01


.. parsed-literal::

      49	 1.1853663e+00	 1.4686357e-01	 1.2334219e+00	 1.5399947e-01	[ 1.1687248e+00]	 2.1836948e-01
      50	 1.1952954e+00	 1.4611407e-01	 1.2434756e+00	 1.5308660e-01	[ 1.1720280e+00]	 2.0308733e-01


.. parsed-literal::

      51	 1.2052259e+00	 1.4502070e-01	 1.2536281e+00	 1.5087861e-01	[ 1.1737629e+00]	 2.1008515e-01


.. parsed-literal::

      52	 1.2153418e+00	 1.4439865e-01	 1.2638557e+00	 1.4960882e-01	[ 1.1795410e+00]	 2.1803498e-01
      53	 1.2264212e+00	 1.4405212e-01	 1.2752270e+00	 1.5001621e-01	[ 1.1862587e+00]	 1.7280149e-01


.. parsed-literal::

      54	 1.2358521e+00	 1.4353000e-01	 1.2847608e+00	 1.5084944e-01	[ 1.1963505e+00]	 2.1171379e-01
      55	 1.2446671e+00	 1.4370877e-01	 1.2935987e+00	 1.5256492e-01	[ 1.2058577e+00]	 1.9761896e-01


.. parsed-literal::

      56	 1.2530949e+00	 1.4385299e-01	 1.3021991e+00	 1.5373559e-01	[ 1.2146309e+00]	 2.1285868e-01


.. parsed-literal::

      57	 1.2622151e+00	 1.4305429e-01	 1.3113089e+00	 1.5332538e-01	[ 1.2245833e+00]	 2.1154475e-01


.. parsed-literal::

      58	 1.2721630e+00	 1.4264803e-01	 1.3213575e+00	 1.5277859e-01	[ 1.2375712e+00]	 2.1067309e-01


.. parsed-literal::

      59	 1.2812955e+00	 1.4159682e-01	 1.3304283e+00	 1.5165179e-01	[ 1.2437722e+00]	 2.0146084e-01
      60	 1.2888480e+00	 1.4118005e-01	 1.3381272e+00	 1.5172838e-01	[ 1.2469870e+00]	 1.8579221e-01


.. parsed-literal::

      61	 1.2944050e+00	 1.4063265e-01	 1.3440088e+00	 1.5242585e-01	  1.2402609e+00 	 2.1606302e-01


.. parsed-literal::

      62	 1.3007641e+00	 1.4028109e-01	 1.3502933e+00	 1.5274192e-01	  1.2460918e+00 	 2.1382070e-01


.. parsed-literal::

      63	 1.3076989e+00	 1.3998176e-01	 1.3573277e+00	 1.5366406e-01	[ 1.2512054e+00]	 2.0406985e-01


.. parsed-literal::

      64	 1.3150807e+00	 1.3950972e-01	 1.3647585e+00	 1.5373366e-01	[ 1.2588291e+00]	 2.0849538e-01


.. parsed-literal::

      65	 1.3238404e+00	 1.3859806e-01	 1.3740594e+00	 1.5395062e-01	[ 1.2639310e+00]	 2.0994449e-01


.. parsed-literal::

      66	 1.3324018e+00	 1.3838984e-01	 1.3825179e+00	 1.5311530e-01	[ 1.2757113e+00]	 2.1328306e-01


.. parsed-literal::

      67	 1.3373985e+00	 1.3810466e-01	 1.3873689e+00	 1.5304143e-01	[ 1.2812844e+00]	 2.1710038e-01
      68	 1.3447786e+00	 1.3731734e-01	 1.3949228e+00	 1.5364914e-01	[ 1.2846810e+00]	 1.9321752e-01


.. parsed-literal::

      69	 1.3502859e+00	 1.3689918e-01	 1.4005543e+00	 1.5463575e-01	[ 1.2910358e+00]	 2.1193314e-01
      70	 1.3563899e+00	 1.3641273e-01	 1.4065891e+00	 1.5531146e-01	[ 1.2973777e+00]	 1.9958353e-01


.. parsed-literal::

      71	 1.3621752e+00	 1.3595221e-01	 1.4123337e+00	 1.5626209e-01	[ 1.2995812e+00]	 2.0864844e-01


.. parsed-literal::

      72	 1.3659109e+00	 1.3554479e-01	 1.4161342e+00	 1.5621805e-01	[ 1.3045717e+00]	 2.0727444e-01
      73	 1.3698717e+00	 1.3529916e-01	 1.4201790e+00	 1.5603900e-01	[ 1.3064048e+00]	 1.7422891e-01


.. parsed-literal::

      74	 1.3797553e+00	 1.3461076e-01	 1.4304919e+00	 1.5617996e-01	[ 1.3103297e+00]	 2.0781732e-01
      75	 1.3820160e+00	 1.3445909e-01	 1.4329677e+00	 1.5604759e-01	  1.3093709e+00 	 1.8467855e-01


.. parsed-literal::

      76	 1.3891884e+00	 1.3407351e-01	 1.4399560e+00	 1.5572039e-01	[ 1.3179112e+00]	 2.0316267e-01


.. parsed-literal::

      77	 1.3930809e+00	 1.3377990e-01	 1.4438044e+00	 1.5568932e-01	[ 1.3217004e+00]	 2.1040249e-01
      78	 1.3971302e+00	 1.3345054e-01	 1.4478724e+00	 1.5553128e-01	[ 1.3235755e+00]	 1.6871142e-01


.. parsed-literal::

      79	 1.4039808e+00	 1.3303477e-01	 1.4549350e+00	 1.5514906e-01	[ 1.3236567e+00]	 2.0487475e-01


.. parsed-literal::

      80	 1.4081475e+00	 1.3259971e-01	 1.4592160e+00	 1.5501985e-01	  1.3220945e+00 	 3.0955291e-01
      81	 1.4136473e+00	 1.3243276e-01	 1.4648498e+00	 1.5434886e-01	  1.3206003e+00 	 1.8333721e-01


.. parsed-literal::

      82	 1.4172753e+00	 1.3227673e-01	 1.4685365e+00	 1.5402178e-01	  1.3220125e+00 	 2.0008707e-01


.. parsed-literal::

      83	 1.4225634e+00	 1.3198060e-01	 1.4739055e+00	 1.5340889e-01	  1.3177459e+00 	 2.0370531e-01
      84	 1.4256869e+00	 1.3135242e-01	 1.4772806e+00	 1.5263489e-01	  1.3226068e+00 	 1.6629887e-01


.. parsed-literal::

      85	 1.4298260e+00	 1.3117943e-01	 1.4812835e+00	 1.5251182e-01	  1.3220734e+00 	 2.0420718e-01
      86	 1.4321181e+00	 1.3092638e-01	 1.4835383e+00	 1.5231864e-01	  1.3227438e+00 	 1.8846869e-01


.. parsed-literal::

      87	 1.4354689e+00	 1.3049831e-01	 1.4869239e+00	 1.5184551e-01	[ 1.3238897e+00]	 1.8476057e-01
      88	 1.4386211e+00	 1.3007585e-01	 1.4902250e+00	 1.5169107e-01	  1.3234216e+00 	 1.9323397e-01


.. parsed-literal::

      89	 1.4424045e+00	 1.2973026e-01	 1.4940630e+00	 1.5096920e-01	[ 1.3262817e+00]	 2.0774841e-01


.. parsed-literal::

      90	 1.4444773e+00	 1.2970906e-01	 1.4961343e+00	 1.5089258e-01	[ 1.3275614e+00]	 2.0879126e-01


.. parsed-literal::

      91	 1.4472954e+00	 1.2964399e-01	 1.4990566e+00	 1.5075612e-01	[ 1.3281621e+00]	 2.0383501e-01


.. parsed-literal::

      92	 1.4495538e+00	 1.2941771e-01	 1.5015895e+00	 1.5114957e-01	  1.3213474e+00 	 2.0106268e-01
      93	 1.4529835e+00	 1.2929627e-01	 1.5049704e+00	 1.5083365e-01	  1.3254661e+00 	 1.7697215e-01


.. parsed-literal::

      94	 1.4554699e+00	 1.2906050e-01	 1.5074615e+00	 1.5062308e-01	[ 1.3285456e+00]	 2.1188211e-01


.. parsed-literal::

      95	 1.4578600e+00	 1.2883466e-01	 1.5098674e+00	 1.5042859e-01	[ 1.3304587e+00]	 2.1327281e-01
      96	 1.4617477e+00	 1.2845307e-01	 1.5138403e+00	 1.5032809e-01	[ 1.3385010e+00]	 1.9393349e-01


.. parsed-literal::

      97	 1.4654301e+00	 1.2829727e-01	 1.5176073e+00	 1.5000176e-01	  1.3342005e+00 	 1.9354296e-01


.. parsed-literal::

      98	 1.4678565e+00	 1.2814486e-01	 1.5200192e+00	 1.4994891e-01	  1.3384133e+00 	 2.0271993e-01


.. parsed-literal::

      99	 1.4709607e+00	 1.2771463e-01	 1.5231732e+00	 1.4961813e-01	[ 1.3406220e+00]	 2.1475482e-01


.. parsed-literal::

     100	 1.4741127e+00	 1.2708252e-01	 1.5263238e+00	 1.4922796e-01	[ 1.3452106e+00]	 2.1057963e-01


.. parsed-literal::

     101	 1.4768974e+00	 1.2649800e-01	 1.5290725e+00	 1.4858842e-01	[ 1.3483525e+00]	 2.1703720e-01


.. parsed-literal::

     102	 1.4784846e+00	 1.2636577e-01	 1.5306523e+00	 1.4845994e-01	  1.3455921e+00 	 2.0358348e-01


.. parsed-literal::

     103	 1.4806366e+00	 1.2632278e-01	 1.5328016e+00	 1.4835859e-01	  1.3442063e+00 	 2.0987105e-01
     104	 1.4841721e+00	 1.2623625e-01	 1.5364159e+00	 1.4802094e-01	  1.3436736e+00 	 1.9444752e-01


.. parsed-literal::

     105	 1.4853341e+00	 1.2655078e-01	 1.5377713e+00	 1.4900747e-01	  1.3345346e+00 	 2.0776486e-01


.. parsed-literal::

     106	 1.4889637e+00	 1.2633781e-01	 1.5412587e+00	 1.4827627e-01	  1.3437085e+00 	 2.0568633e-01
     107	 1.4905870e+00	 1.2624273e-01	 1.5428574e+00	 1.4810747e-01	  1.3465232e+00 	 1.6717267e-01


.. parsed-literal::

     108	 1.4935127e+00	 1.2614614e-01	 1.5458094e+00	 1.4805022e-01	  1.3477797e+00 	 1.9549060e-01
     109	 1.4964455e+00	 1.2608120e-01	 1.5488537e+00	 1.4843854e-01	  1.3437893e+00 	 1.8676209e-01


.. parsed-literal::

     110	 1.4987813e+00	 1.2604201e-01	 1.5512674e+00	 1.4882564e-01	  1.3397997e+00 	 2.1498537e-01
     111	 1.5003506e+00	 1.2598797e-01	 1.5528039e+00	 1.4899235e-01	  1.3410190e+00 	 1.8687773e-01


.. parsed-literal::

     112	 1.5019055e+00	 1.2593145e-01	 1.5543825e+00	 1.4923174e-01	  1.3389803e+00 	 2.0737696e-01


.. parsed-literal::

     113	 1.5035104e+00	 1.2587149e-01	 1.5560395e+00	 1.4966684e-01	  1.3398416e+00 	 2.1477962e-01


.. parsed-literal::

     114	 1.5055967e+00	 1.2573208e-01	 1.5581441e+00	 1.4980011e-01	  1.3377175e+00 	 2.1197438e-01


.. parsed-literal::

     115	 1.5076608e+00	 1.2580751e-01	 1.5602311e+00	 1.5041135e-01	  1.3391717e+00 	 2.1300316e-01
     116	 1.5097293e+00	 1.2571426e-01	 1.5622480e+00	 1.5024628e-01	  1.3408368e+00 	 1.9881368e-01


.. parsed-literal::

     117	 1.5115889e+00	 1.2571769e-01	 1.5640755e+00	 1.5038450e-01	  1.3432651e+00 	 1.7913842e-01


.. parsed-literal::

     118	 1.5133285e+00	 1.2568666e-01	 1.5658199e+00	 1.5016560e-01	  1.3421932e+00 	 2.2131467e-01
     119	 1.5150428e+00	 1.2565054e-01	 1.5675786e+00	 1.5037526e-01	  1.3412323e+00 	 2.0453000e-01


.. parsed-literal::

     120	 1.5165950e+00	 1.2554881e-01	 1.5691942e+00	 1.5025623e-01	  1.3424811e+00 	 1.7090940e-01


.. parsed-literal::

     121	 1.5181304e+00	 1.2532790e-01	 1.5708059e+00	 1.4984547e-01	  1.3404331e+00 	 2.0802188e-01


.. parsed-literal::

     122	 1.5205160e+00	 1.2494909e-01	 1.5733195e+00	 1.4868126e-01	  1.3437017e+00 	 2.0828247e-01


.. parsed-literal::

     123	 1.5224698e+00	 1.2468243e-01	 1.5753130e+00	 1.4788490e-01	  1.3425113e+00 	 2.1175146e-01


.. parsed-literal::

     124	 1.5242724e+00	 1.2460706e-01	 1.5770810e+00	 1.4747651e-01	  1.3399609e+00 	 2.2045255e-01
     125	 1.5257999e+00	 1.2457832e-01	 1.5786159e+00	 1.4717933e-01	  1.3397832e+00 	 2.0823407e-01


.. parsed-literal::

     126	 1.5270304e+00	 1.2447803e-01	 1.5799413e+00	 1.4673817e-01	  1.3307820e+00 	 2.0937467e-01


.. parsed-literal::

     127	 1.5284617e+00	 1.2441878e-01	 1.5814130e+00	 1.4663872e-01	  1.3295629e+00 	 2.2126579e-01


.. parsed-literal::

     128	 1.5303939e+00	 1.2428609e-01	 1.5834396e+00	 1.4655411e-01	  1.3268611e+00 	 2.0368266e-01


.. parsed-literal::

     129	 1.5319858e+00	 1.2415919e-01	 1.5850788e+00	 1.4639054e-01	  1.3246569e+00 	 2.1470094e-01


.. parsed-literal::

     130	 1.5328054e+00	 1.2362828e-01	 1.5861279e+00	 1.4506768e-01	  1.3186433e+00 	 2.0616770e-01
     131	 1.5354269e+00	 1.2369597e-01	 1.5886006e+00	 1.4547281e-01	  1.3223963e+00 	 1.9905448e-01


.. parsed-literal::

     132	 1.5361794e+00	 1.2365725e-01	 1.5893315e+00	 1.4524005e-01	  1.3231344e+00 	 1.9863248e-01
     133	 1.5378136e+00	 1.2343494e-01	 1.5910134e+00	 1.4465263e-01	  1.3216572e+00 	 1.8363190e-01


.. parsed-literal::

     134	 1.5387382e+00	 1.2321908e-01	 1.5920639e+00	 1.4385313e-01	  1.3187975e+00 	 2.0605874e-01


.. parsed-literal::

     135	 1.5402831e+00	 1.2316928e-01	 1.5935895e+00	 1.4400903e-01	  1.3175020e+00 	 2.0384693e-01
     136	 1.5415328e+00	 1.2307545e-01	 1.5948777e+00	 1.4418386e-01	  1.3142085e+00 	 2.0065951e-01


.. parsed-literal::

     137	 1.5426537e+00	 1.2298345e-01	 1.5960294e+00	 1.4433311e-01	  1.3117558e+00 	 2.1883297e-01
     138	 1.5449714e+00	 1.2279334e-01	 1.5984054e+00	 1.4439818e-01	  1.3090885e+00 	 1.8651128e-01


.. parsed-literal::

     139	 1.5461767e+00	 1.2269145e-01	 1.5996736e+00	 1.4443053e-01	  1.3076063e+00 	 3.2953572e-01


.. parsed-literal::

     140	 1.5479765e+00	 1.2257585e-01	 1.6014858e+00	 1.4434801e-01	  1.3084386e+00 	 2.1612048e-01


.. parsed-literal::

     141	 1.5496250e+00	 1.2248372e-01	 1.6031369e+00	 1.4407600e-01	  1.3085015e+00 	 2.1503782e-01


.. parsed-literal::

     142	 1.5508027e+00	 1.2229007e-01	 1.6043450e+00	 1.4372172e-01	  1.3059961e+00 	 2.1414781e-01


.. parsed-literal::

     143	 1.5521609e+00	 1.2231241e-01	 1.6056566e+00	 1.4359280e-01	  1.3079535e+00 	 2.1542215e-01


.. parsed-literal::

     144	 1.5529118e+00	 1.2227853e-01	 1.6063916e+00	 1.4347997e-01	  1.3069732e+00 	 2.0744634e-01


.. parsed-literal::

     145	 1.5544011e+00	 1.2215532e-01	 1.6078650e+00	 1.4308457e-01	  1.3063571e+00 	 2.1430349e-01


.. parsed-literal::

     146	 1.5561066e+00	 1.2202594e-01	 1.6095649e+00	 1.4241551e-01	  1.3053337e+00 	 2.0956326e-01


.. parsed-literal::

     147	 1.5568316e+00	 1.2191026e-01	 1.6103017e+00	 1.4154359e-01	  1.3144317e+00 	 2.0608306e-01


.. parsed-literal::

     148	 1.5589529e+00	 1.2190229e-01	 1.6123878e+00	 1.4129492e-01	  1.3086907e+00 	 2.1113253e-01


.. parsed-literal::

     149	 1.5595632e+00	 1.2193006e-01	 1.6129837e+00	 1.4124842e-01	  1.3085843e+00 	 2.0651150e-01


.. parsed-literal::

     150	 1.5604537e+00	 1.2196776e-01	 1.6138740e+00	 1.4109691e-01	  1.3086325e+00 	 2.1543550e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe004b7eaa0>



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


.. parsed-literal::

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
    CPU times: user 2.06 s, sys: 52 ms, total: 2.11 s
    Wall time: 638 ms


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

