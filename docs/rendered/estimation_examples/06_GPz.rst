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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5235283e-01	 3.2296748e-01	-3.4271450e-01	 3.1050976e-01	[-3.2015817e-01]	 4.6096230e-01


.. parsed-literal::

       2	-2.8027832e-01	 3.1229049e-01	-2.5613443e-01	 2.9995748e-01	[-2.1982291e-01]	 2.2557735e-01


.. parsed-literal::

       3	-2.3271030e-01	 2.8936726e-01	-1.8799265e-01	 2.7943736e-01	[-1.4838187e-01]	 2.8521633e-01
       4	-2.0638093e-01	 2.6654518e-01	-1.6620834e-01	 2.6243888e-01	[-1.3900284e-01]	 1.7440128e-01


.. parsed-literal::

       5	-1.0748265e-01	 2.5862492e-01	-7.2189197e-02	 2.5237259e-01	[-4.4067268e-02]	 2.0962453e-01


.. parsed-literal::

       6	-7.7371855e-02	 2.5341476e-01	-4.6727327e-02	 2.4605788e-01	[-2.0629200e-02]	 2.0514536e-01
       7	-5.6177793e-02	 2.4995332e-01	-3.2962349e-02	 2.4350237e-01	[-7.8158662e-03]	 1.9535851e-01


.. parsed-literal::

       8	-4.5474420e-02	 2.4816632e-01	-2.5498946e-02	 2.4254221e-01	[-2.7220862e-03]	 2.0359635e-01
       9	-3.2224327e-02	 2.4565373e-01	-1.4940388e-02	 2.4145791e-01	[ 3.0294576e-03]	 1.7985058e-01


.. parsed-literal::

      10	-2.2244062e-02	 2.4362094e-01	-6.6659128e-03	 2.3904341e-01	[ 1.5064744e-02]	 2.0494652e-01


.. parsed-literal::

      11	-1.7843455e-02	 2.4306188e-01	-3.4813449e-03	 2.3756007e-01	[ 2.0048528e-02]	 2.0482469e-01


.. parsed-literal::

      12	-1.4400928e-02	 2.4238162e-01	-2.8725348e-04	 2.3688365e-01	[ 2.3847242e-02]	 2.0794320e-01
      13	-1.0794506e-02	 2.4165896e-01	 3.2850274e-03	 2.3565044e-01	[ 2.9902056e-02]	 1.7466474e-01


.. parsed-literal::

      14	-4.9256993e-03	 2.4036010e-01	 1.0005497e-02	 2.3354300e-01	[ 3.9719663e-02]	 2.0624900e-01
      15	 9.8917914e-02	 2.2750786e-01	 1.1901614e-01	 2.1566057e-01	[ 1.5518874e-01]	 1.9915533e-01


.. parsed-literal::

      16	 1.4846858e-01	 2.2467284e-01	 1.7316467e-01	 2.1587972e-01	[ 1.8598316e-01]	 1.9944167e-01
      17	 2.7866078e-01	 2.2028023e-01	 3.0842826e-01	 2.1066417e-01	[ 3.3056106e-01]	 1.6902184e-01


.. parsed-literal::

      18	 3.3416118e-01	 2.1716331e-01	 3.6847821e-01	 2.0859381e-01	[ 3.8320004e-01]	 2.0663071e-01


.. parsed-literal::

      19	 3.8386966e-01	 2.1199751e-01	 4.1816695e-01	 2.0449167e-01	[ 4.2766653e-01]	 2.1111822e-01
      20	 4.2995590e-01	 2.0814726e-01	 4.6429196e-01	 1.9915271e-01	[ 4.7241273e-01]	 2.0238185e-01


.. parsed-literal::

      21	 5.0366185e-01	 2.0417569e-01	 5.3872214e-01	 1.9568495e-01	[ 5.3322164e-01]	 1.9823456e-01


.. parsed-literal::

      22	 6.0146708e-01	 2.0272826e-01	 6.4067974e-01	 1.9287453e-01	[ 5.7735508e-01]	 2.1140599e-01


.. parsed-literal::

      23	 6.3885427e-01	 1.9904567e-01	 6.7774436e-01	 1.8867776e-01	[ 6.1485883e-01]	 2.0288372e-01
      24	 6.6968085e-01	 1.9767877e-01	 7.0851242e-01	 1.8754707e-01	[ 6.4474386e-01]	 1.9876599e-01


.. parsed-literal::

      25	 7.1509197e-01	 1.9894352e-01	 7.5320148e-01	 1.8935932e-01	[ 6.9811661e-01]	 2.0287371e-01
      26	 7.4183588e-01	 2.0400357e-01	 7.7924640e-01	 1.9310957e-01	[ 7.3676809e-01]	 1.8516445e-01


.. parsed-literal::

      27	 7.9468997e-01	 2.0139514e-01	 8.3375372e-01	 1.8836093e-01	[ 7.8039210e-01]	 1.8958950e-01


.. parsed-literal::

      28	 8.3456435e-01	 1.9965253e-01	 8.7538464e-01	 1.8667245e-01	[ 8.1987713e-01]	 2.1467233e-01
      29	 8.5823865e-01	 1.9762926e-01	 9.0013436e-01	 1.8621780e-01	[ 8.3645273e-01]	 1.7432046e-01


.. parsed-literal::

      30	 8.7995195e-01	 1.9531470e-01	 9.2147443e-01	 1.8423170e-01	[ 8.6548480e-01]	 2.1796560e-01
      31	 9.0199292e-01	 1.9127245e-01	 9.4480399e-01	 1.8343874e-01	[ 8.6748773e-01]	 1.7692685e-01


.. parsed-literal::

      32	 9.2157378e-01	 1.8951021e-01	 9.6484849e-01	 1.8196386e-01	[ 8.9044515e-01]	 2.0026684e-01


.. parsed-literal::

      33	 9.4202145e-01	 1.8618974e-01	 9.8609597e-01	 1.7883157e-01	[ 9.1296363e-01]	 2.1125507e-01


.. parsed-literal::

      34	 9.7059450e-01	 1.8076342e-01	 1.0152563e+00	 1.7450760e-01	[ 9.4765268e-01]	 2.1256995e-01
      35	 9.9218315e-01	 1.7802786e-01	 1.0373384e+00	 1.7025823e-01	[ 9.8085806e-01]	 1.9514012e-01


.. parsed-literal::

      36	 1.0097989e+00	 1.7443611e-01	 1.0552059e+00	 1.6585037e-01	[ 1.0013165e+00]	 2.0171046e-01


.. parsed-literal::

      37	 1.0260997e+00	 1.7082950e-01	 1.0717622e+00	 1.6164424e-01	[ 1.0142695e+00]	 2.1629310e-01


.. parsed-literal::

      38	 1.0330287e+00	 1.6871209e-01	 1.0801490e+00	 1.5703395e-01	[ 1.0193888e+00]	 2.0381665e-01


.. parsed-literal::

      39	 1.0480038e+00	 1.6864406e-01	 1.0946590e+00	 1.5812080e-01	[ 1.0286620e+00]	 2.1364045e-01


.. parsed-literal::

      40	 1.0545373e+00	 1.6850402e-01	 1.1012263e+00	 1.5841421e-01	[ 1.0302342e+00]	 2.1237564e-01
      41	 1.0653624e+00	 1.6756359e-01	 1.1122350e+00	 1.5742339e-01	[ 1.0373450e+00]	 1.9354701e-01


.. parsed-literal::

      42	 1.0816728e+00	 1.6588613e-01	 1.1286419e+00	 1.5508724e-01	[ 1.0552892e+00]	 2.0089936e-01


.. parsed-literal::

      43	 1.0907651e+00	 1.6357459e-01	 1.1382343e+00	 1.5114068e-01	[ 1.0724733e+00]	 3.1402159e-01
      44	 1.1013344e+00	 1.6190698e-01	 1.1486796e+00	 1.4879526e-01	[ 1.0871118e+00]	 1.8042874e-01


.. parsed-literal::

      45	 1.1161167e+00	 1.5861889e-01	 1.1635512e+00	 1.4509691e-01	[ 1.1034055e+00]	 2.0120692e-01
      46	 1.1296868e+00	 1.5572746e-01	 1.1770828e+00	 1.4180388e-01	[ 1.1279588e+00]	 1.9882035e-01


.. parsed-literal::

      47	 1.1427010e+00	 1.5304632e-01	 1.1907249e+00	 1.3943082e-01	[ 1.1390606e+00]	 2.0523524e-01
      48	 1.1546456e+00	 1.5159893e-01	 1.2025890e+00	 1.3823521e-01	[ 1.1470622e+00]	 1.9754672e-01


.. parsed-literal::

      49	 1.1695696e+00	 1.5004840e-01	 1.2178349e+00	 1.3745773e-01	[ 1.1566347e+00]	 1.9672012e-01


.. parsed-literal::

      50	 1.1832559e+00	 1.4862808e-01	 1.2316589e+00	 1.3635120e-01	[ 1.1684343e+00]	 2.0222807e-01


.. parsed-literal::

      51	 1.1962972e+00	 1.4726323e-01	 1.2449606e+00	 1.3552404e-01	[ 1.1757596e+00]	 2.1362305e-01


.. parsed-literal::

      52	 1.2080225e+00	 1.4651775e-01	 1.2569903e+00	 1.3514107e-01	[ 1.1820215e+00]	 2.0276427e-01


.. parsed-literal::

      53	 1.2253197e+00	 1.4470942e-01	 1.2752562e+00	 1.3397846e-01	[ 1.1977824e+00]	 2.0928311e-01


.. parsed-literal::

      54	 1.2375405e+00	 1.4532929e-01	 1.2875877e+00	 1.3506307e-01	[ 1.1982781e+00]	 2.0800781e-01
      55	 1.2487769e+00	 1.4383773e-01	 1.2986627e+00	 1.3322980e-01	[ 1.2119851e+00]	 1.7572403e-01


.. parsed-literal::

      56	 1.2594927e+00	 1.4231304e-01	 1.3096697e+00	 1.3132986e-01	[ 1.2235253e+00]	 2.0639253e-01
      57	 1.2688076e+00	 1.4215258e-01	 1.3189813e+00	 1.3078846e-01	[ 1.2354173e+00]	 1.8361950e-01


.. parsed-literal::

      58	 1.2802065e+00	 1.4103562e-01	 1.3305095e+00	 1.2922922e-01	[ 1.2487362e+00]	 2.0162225e-01


.. parsed-literal::

      59	 1.2916522e+00	 1.4057693e-01	 1.3423348e+00	 1.2877584e-01	[ 1.2625993e+00]	 2.0441914e-01
      60	 1.3025457e+00	 1.3995494e-01	 1.3536224e+00	 1.2749603e-01	[ 1.2712881e+00]	 1.7980909e-01


.. parsed-literal::

      61	 1.3112463e+00	 1.3926167e-01	 1.3625091e+00	 1.2655699e-01	[ 1.2737755e+00]	 2.0553541e-01


.. parsed-literal::

      62	 1.3195169e+00	 1.3877558e-01	 1.3705763e+00	 1.2554558e-01	[ 1.2815416e+00]	 2.0733118e-01
      63	 1.3273113e+00	 1.3826648e-01	 1.3783758e+00	 1.2417769e-01	  1.2810631e+00 	 1.9673443e-01


.. parsed-literal::

      64	 1.3364759e+00	 1.3738368e-01	 1.3878831e+00	 1.2249846e-01	[ 1.2843832e+00]	 2.1249413e-01


.. parsed-literal::

      65	 1.3457510e+00	 1.3580204e-01	 1.3971602e+00	 1.2012644e-01	[ 1.2925257e+00]	 2.2183037e-01


.. parsed-literal::

      66	 1.3544013e+00	 1.3481346e-01	 1.4058814e+00	 1.1977816e-01	[ 1.3096252e+00]	 2.1438932e-01
      67	 1.3604376e+00	 1.3425494e-01	 1.4119868e+00	 1.1992048e-01	[ 1.3215363e+00]	 1.9972754e-01


.. parsed-literal::

      68	 1.3676616e+00	 1.3363979e-01	 1.4194198e+00	 1.2006658e-01	[ 1.3294912e+00]	 2.0472479e-01


.. parsed-literal::

      69	 1.3729653e+00	 1.3289438e-01	 1.4249418e+00	 1.1986841e-01	[ 1.3374242e+00]	 2.1155548e-01
      70	 1.3805503e+00	 1.3254751e-01	 1.4324669e+00	 1.1926051e-01	[ 1.3403175e+00]	 2.0142531e-01


.. parsed-literal::

      71	 1.3864902e+00	 1.3225817e-01	 1.4385349e+00	 1.1901986e-01	[ 1.3405260e+00]	 2.1694708e-01
      72	 1.3931588e+00	 1.3188034e-01	 1.4454472e+00	 1.1875328e-01	[ 1.3410663e+00]	 1.7443085e-01


.. parsed-literal::

      73	 1.3970382e+00	 1.3117770e-01	 1.4496602e+00	 1.1992344e-01	[ 1.3451316e+00]	 2.0080471e-01


.. parsed-literal::

      74	 1.4050746e+00	 1.3108651e-01	 1.4575751e+00	 1.1946783e-01	[ 1.3539950e+00]	 2.0878649e-01
      75	 1.4089400e+00	 1.3105444e-01	 1.4613461e+00	 1.1950167e-01	[ 1.3588202e+00]	 1.8443942e-01


.. parsed-literal::

      76	 1.4145251e+00	 1.3095425e-01	 1.4669876e+00	 1.1981808e-01	[ 1.3626580e+00]	 2.1026254e-01


.. parsed-literal::

      77	 1.4204359e+00	 1.3098130e-01	 1.4727392e+00	 1.1991056e-01	[ 1.3662144e+00]	 2.1432996e-01
      78	 1.4255736e+00	 1.3084353e-01	 1.4778125e+00	 1.1999568e-01	  1.3640977e+00 	 1.7744851e-01


.. parsed-literal::

      79	 1.4292931e+00	 1.3055414e-01	 1.4815498e+00	 1.1963734e-01	  1.3661510e+00 	 2.0762396e-01


.. parsed-literal::

      80	 1.4354196e+00	 1.2975766e-01	 1.4878931e+00	 1.1888443e-01	  1.3645327e+00 	 2.0787048e-01
      81	 1.4393042e+00	 1.2935109e-01	 1.4918229e+00	 1.1828290e-01	[ 1.3680815e+00]	 1.8415022e-01


.. parsed-literal::

      82	 1.4437124e+00	 1.2897836e-01	 1.4961893e+00	 1.1822980e-01	[ 1.3728327e+00]	 1.9559979e-01
      83	 1.4472537e+00	 1.2878076e-01	 1.4996914e+00	 1.1825773e-01	[ 1.3741567e+00]	 1.9649863e-01


.. parsed-literal::

      84	 1.4503137e+00	 1.2868770e-01	 1.5028488e+00	 1.1842171e-01	[ 1.3749261e+00]	 2.0670533e-01


.. parsed-literal::

      85	 1.4545212e+00	 1.2883678e-01	 1.5073766e+00	 1.1881046e-01	  1.3664822e+00 	 2.0949912e-01


.. parsed-literal::

      86	 1.4589132e+00	 1.2855617e-01	 1.5118379e+00	 1.1920762e-01	  1.3687599e+00 	 2.0703411e-01


.. parsed-literal::

      87	 1.4618329e+00	 1.2849709e-01	 1.5147623e+00	 1.1927248e-01	  1.3688610e+00 	 2.1444368e-01
      88	 1.4661192e+00	 1.2826007e-01	 1.5191920e+00	 1.1931601e-01	  1.3635150e+00 	 1.9912934e-01


.. parsed-literal::

      89	 1.4690395e+00	 1.2830221e-01	 1.5223984e+00	 1.1897859e-01	  1.3584565e+00 	 2.0600605e-01


.. parsed-literal::

      90	 1.4733243e+00	 1.2793210e-01	 1.5266028e+00	 1.1849011e-01	  1.3594803e+00 	 2.2211194e-01


.. parsed-literal::

      91	 1.4761049e+00	 1.2772289e-01	 1.5294110e+00	 1.1810129e-01	  1.3591654e+00 	 2.0759392e-01


.. parsed-literal::

      92	 1.4790315e+00	 1.2750214e-01	 1.5324229e+00	 1.1780211e-01	  1.3582486e+00 	 2.1371269e-01
      93	 1.4824608e+00	 1.2720910e-01	 1.5361179e+00	 1.1752154e-01	  1.3506442e+00 	 1.7580533e-01


.. parsed-literal::

      94	 1.4869379e+00	 1.2681867e-01	 1.5406200e+00	 1.1726901e-01	  1.3547410e+00 	 1.7580366e-01


.. parsed-literal::

      95	 1.4892358e+00	 1.2665480e-01	 1.5428486e+00	 1.1739689e-01	  1.3568067e+00 	 2.1460724e-01
      96	 1.4926945e+00	 1.2628104e-01	 1.5463327e+00	 1.1745177e-01	  1.3561900e+00 	 1.9297910e-01


.. parsed-literal::

      97	 1.4945170e+00	 1.2600135e-01	 1.5482019e+00	 1.1766313e-01	  1.3517650e+00 	 2.1029782e-01
      98	 1.4974604e+00	 1.2570105e-01	 1.5510985e+00	 1.1750346e-01	  1.3532601e+00 	 1.6915464e-01


.. parsed-literal::

      99	 1.4993551e+00	 1.2553075e-01	 1.5529556e+00	 1.1730737e-01	  1.3538402e+00 	 2.2794199e-01


.. parsed-literal::

     100	 1.5015960e+00	 1.2523543e-01	 1.5551659e+00	 1.1717455e-01	  1.3532641e+00 	 2.0343876e-01
     101	 1.5047896e+00	 1.2487028e-01	 1.5583924e+00	 1.1718696e-01	  1.3507656e+00 	 2.0347929e-01


.. parsed-literal::

     102	 1.5074210e+00	 1.2418136e-01	 1.5610666e+00	 1.1761179e-01	  1.3468740e+00 	 1.9174981e-01


.. parsed-literal::

     103	 1.5094023e+00	 1.2412958e-01	 1.5630265e+00	 1.1772698e-01	  1.3478219e+00 	 2.0838737e-01
     104	 1.5115264e+00	 1.2400785e-01	 1.5651949e+00	 1.1795653e-01	  1.3477071e+00 	 1.9237137e-01


.. parsed-literal::

     105	 1.5136080e+00	 1.2386123e-01	 1.5673202e+00	 1.1803128e-01	  1.3458922e+00 	 2.1443701e-01


.. parsed-literal::

     106	 1.5156931e+00	 1.2367960e-01	 1.5695093e+00	 1.1810630e-01	  1.3451233e+00 	 2.0397472e-01
     107	 1.5174188e+00	 1.2359648e-01	 1.5711829e+00	 1.1801624e-01	  1.3433897e+00 	 2.0150089e-01


.. parsed-literal::

     108	 1.5184027e+00	 1.2352809e-01	 1.5721509e+00	 1.1789011e-01	  1.3436219e+00 	 2.0086503e-01


.. parsed-literal::

     109	 1.5206081e+00	 1.2335822e-01	 1.5744153e+00	 1.1779991e-01	  1.3444043e+00 	 2.0516729e-01


.. parsed-literal::

     110	 1.5222977e+00	 1.2327766e-01	 1.5762224e+00	 1.1768094e-01	  1.3441381e+00 	 3.3044624e-01
     111	 1.5243376e+00	 1.2314183e-01	 1.5783768e+00	 1.1787920e-01	  1.3448353e+00 	 1.6983628e-01


.. parsed-literal::

     112	 1.5261981e+00	 1.2306987e-01	 1.5803352e+00	 1.1811919e-01	  1.3458559e+00 	 1.7108130e-01
     113	 1.5279446e+00	 1.2303577e-01	 1.5821768e+00	 1.1843321e-01	  1.3427295e+00 	 1.8882585e-01


.. parsed-literal::

     114	 1.5295817e+00	 1.2302187e-01	 1.5838322e+00	 1.1869086e-01	  1.3415388e+00 	 2.1251130e-01
     115	 1.5310347e+00	 1.2297375e-01	 1.5852549e+00	 1.1877900e-01	  1.3379861e+00 	 1.8574762e-01


.. parsed-literal::

     116	 1.5333107e+00	 1.2285206e-01	 1.5875306e+00	 1.1896145e-01	  1.3326929e+00 	 2.0649290e-01
     117	 1.5350241e+00	 1.2280413e-01	 1.5892938e+00	 1.1919923e-01	  1.3283091e+00 	 1.9930387e-01


.. parsed-literal::

     118	 1.5370401e+00	 1.2262565e-01	 1.5913840e+00	 1.1945889e-01	  1.3297305e+00 	 1.9916081e-01
     119	 1.5388931e+00	 1.2249208e-01	 1.5933015e+00	 1.1949262e-01	  1.3324217e+00 	 1.8606591e-01


.. parsed-literal::

     120	 1.5403821e+00	 1.2238098e-01	 1.5948436e+00	 1.1949631e-01	  1.3321480e+00 	 2.1658993e-01


.. parsed-literal::

     121	 1.5422195e+00	 1.2232018e-01	 1.5967530e+00	 1.1928799e-01	  1.3311670e+00 	 2.0686913e-01


.. parsed-literal::

     122	 1.5438697e+00	 1.2223530e-01	 1.5983730e+00	 1.1906073e-01	  1.3251599e+00 	 2.1053743e-01
     123	 1.5450338e+00	 1.2219572e-01	 1.5994949e+00	 1.1898573e-01	  1.3249544e+00 	 2.0178294e-01


.. parsed-literal::

     124	 1.5473078e+00	 1.2207300e-01	 1.6017354e+00	 1.1863167e-01	  1.3246113e+00 	 2.0738983e-01


.. parsed-literal::

     125	 1.5480179e+00	 1.2191428e-01	 1.6024857e+00	 1.1818217e-01	  1.3287417e+00 	 2.0666432e-01
     126	 1.5493562e+00	 1.2186803e-01	 1.6037999e+00	 1.1812196e-01	  1.3291332e+00 	 1.9904470e-01


.. parsed-literal::

     127	 1.5502035e+00	 1.2177649e-01	 1.6046924e+00	 1.1801376e-01	  1.3288052e+00 	 2.0293283e-01


.. parsed-literal::

     128	 1.5514523e+00	 1.2160497e-01	 1.6060323e+00	 1.1775756e-01	  1.3280532e+00 	 2.1337557e-01
     129	 1.5528280e+00	 1.2132301e-01	 1.6076096e+00	 1.1751898e-01	  1.3205202e+00 	 1.8036199e-01


.. parsed-literal::

     130	 1.5547065e+00	 1.2114992e-01	 1.6094961e+00	 1.1723205e-01	  1.3214533e+00 	 2.1116257e-01
     131	 1.5559082e+00	 1.2105523e-01	 1.6106739e+00	 1.1715334e-01	  1.3200086e+00 	 2.0309138e-01


.. parsed-literal::

     132	 1.5572239e+00	 1.2096155e-01	 1.6119586e+00	 1.1710500e-01	  1.3180619e+00 	 2.1327543e-01


.. parsed-literal::

     133	 1.5576324e+00	 1.2090172e-01	 1.6123974e+00	 1.1698347e-01	  1.3131133e+00 	 2.1404219e-01
     134	 1.5594187e+00	 1.2084853e-01	 1.6141185e+00	 1.1703080e-01	  1.3148247e+00 	 1.7989588e-01


.. parsed-literal::

     135	 1.5602187e+00	 1.2084323e-01	 1.6148979e+00	 1.1705464e-01	  1.3165145e+00 	 2.1425533e-01
     136	 1.5612679e+00	 1.2083108e-01	 1.6159681e+00	 1.1705868e-01	  1.3170562e+00 	 1.7058611e-01


.. parsed-literal::

     137	 1.5627333e+00	 1.2084261e-01	 1.6174754e+00	 1.1714209e-01	  1.3148316e+00 	 2.0701003e-01
     138	 1.5640466e+00	 1.2097650e-01	 1.6189560e+00	 1.1710753e-01	  1.3121039e+00 	 2.0099473e-01


.. parsed-literal::

     139	 1.5654851e+00	 1.2092852e-01	 1.6203453e+00	 1.1712567e-01	  1.3095254e+00 	 2.0502520e-01


.. parsed-literal::

     140	 1.5662621e+00	 1.2091869e-01	 1.6211151e+00	 1.1708167e-01	  1.3080456e+00 	 2.0745873e-01


.. parsed-literal::

     141	 1.5673189e+00	 1.2094918e-01	 1.6221938e+00	 1.1701564e-01	  1.3059634e+00 	 2.1318197e-01


.. parsed-literal::

     142	 1.5682022e+00	 1.2092743e-01	 1.6231369e+00	 1.1693682e-01	  1.3042190e+00 	 3.1561995e-01
     143	 1.5694253e+00	 1.2096419e-01	 1.6243795e+00	 1.1682970e-01	  1.3020057e+00 	 2.0068884e-01


.. parsed-literal::

     144	 1.5702995e+00	 1.2094695e-01	 1.6252741e+00	 1.1678198e-01	  1.3006288e+00 	 2.1324754e-01


.. parsed-literal::

     145	 1.5714640e+00	 1.2086089e-01	 1.6264504e+00	 1.1666316e-01	  1.2971923e+00 	 2.0236707e-01
     146	 1.5726809e+00	 1.2076564e-01	 1.6277173e+00	 1.1672701e-01	  1.2916992e+00 	 1.9868255e-01


.. parsed-literal::

     147	 1.5739883e+00	 1.2068481e-01	 1.6289896e+00	 1.1658995e-01	  1.2876979e+00 	 1.9596934e-01
     148	 1.5752205e+00	 1.2065422e-01	 1.6302356e+00	 1.1657668e-01	  1.2822085e+00 	 1.9832277e-01


.. parsed-literal::

     149	 1.5761563e+00	 1.2068072e-01	 1.6312007e+00	 1.1650982e-01	  1.2795538e+00 	 2.0743895e-01
     150	 1.5771082e+00	 1.2071633e-01	 1.6321906e+00	 1.1653685e-01	  1.2755281e+00 	 2.0372963e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 940 ms, total: 2min 3s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f534488d480>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.72 s, sys: 40.9 ms, total: 1.76 s
    Wall time: 538 ms


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

