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
       1	-3.3570210e-01	 3.1821033e-01	-3.2594353e-01	 3.3058320e-01	[-3.4961047e-01]	 4.6728134e-01


.. parsed-literal::

       2	-2.6285158e-01	 3.0650882e-01	-2.3785320e-01	 3.1961564e-01	[-2.7930471e-01]	 2.3176885e-01


.. parsed-literal::

       3	-2.2103984e-01	 2.8811135e-01	-1.8021803e-01	 3.0248247e-01	[-2.4186026e-01]	 2.7819467e-01
       4	-1.8920396e-01	 2.6377346e-01	-1.4743562e-01	 2.7881533e-01	[-2.3819767e-01]	 1.6839337e-01


.. parsed-literal::

       5	-1.0533413e-01	 2.5640355e-01	-6.8383971e-02	 2.7056167e-01	[-1.3871870e-01]	 2.1630335e-01


.. parsed-literal::

       6	-6.4533730e-02	 2.4925403e-01	-3.0728882e-02	 2.6286008e-01	[-8.4311231e-02]	 2.1624303e-01


.. parsed-literal::

       7	-4.7322404e-02	 2.4690892e-01	-2.0440921e-02	 2.6006039e-01	[-7.5098125e-02]	 2.1171284e-01
       8	-2.9123000e-02	 2.4377847e-01	-7.3663819e-03	 2.5676683e-01	[-6.1687197e-02]	 2.1056819e-01


.. parsed-literal::

       9	-1.3006970e-02	 2.4083217e-01	 4.9527614e-03	 2.5495928e-01	[-5.5963988e-02]	 2.0996475e-01


.. parsed-literal::

      10	-5.0499324e-03	 2.3982408e-01	 9.9413515e-03	 2.5425918e-01	[-4.6735283e-02]	 2.0789099e-01
      11	 1.5167018e-03	 2.3845388e-01	 1.5946366e-02	 2.5237639e-01	[-4.2230042e-02]	 1.9097090e-01


.. parsed-literal::

      12	 4.5400757e-03	 2.3792963e-01	 1.8550242e-02	 2.5201295e-01	[-4.1107003e-02]	 2.0843101e-01


.. parsed-literal::

      13	 7.8608632e-03	 2.3733275e-01	 2.1730827e-02	 2.5125384e-01	[-3.7301759e-02]	 2.1635127e-01


.. parsed-literal::

      14	 1.8446001e-02	 2.3550441e-01	 3.3214159e-02	 2.4925377e-01	[-2.5400792e-02]	 2.1523786e-01
      15	 9.7448535e-02	 2.2670443e-01	 1.1598093e-01	 2.4049210e-01	[ 6.3674578e-02]	 1.9894958e-01


.. parsed-literal::

      16	 1.4970232e-01	 2.2498849e-01	 1.7159188e-01	 2.4044969e-01	[ 1.2598205e-01]	 3.2432866e-01


.. parsed-literal::

      17	 2.3085067e-01	 2.2348159e-01	 2.5593253e-01	 2.3853323e-01	[ 2.0080308e-01]	 2.0831060e-01
      18	 2.9867445e-01	 2.1778611e-01	 3.3004853e-01	 2.3457027e-01	[ 2.5867115e-01]	 1.8332171e-01


.. parsed-literal::

      19	 3.4308556e-01	 2.1344460e-01	 3.7601495e-01	 2.2955212e-01	[ 2.9419282e-01]	 1.9439673e-01


.. parsed-literal::

      20	 3.9670838e-01	 2.0839421e-01	 4.2936124e-01	 2.2468071e-01	[ 3.4410830e-01]	 2.0640206e-01


.. parsed-literal::

      21	 4.6041572e-01	 2.0428984e-01	 4.9351911e-01	 2.1908759e-01	[ 4.1463848e-01]	 2.1787167e-01


.. parsed-literal::

      22	 5.8608458e-01	 2.0142496e-01	 6.2173833e-01	 2.1355712e-01	[ 5.5897272e-01]	 2.0484185e-01


.. parsed-literal::

      23	 6.3554237e-01	 2.0130445e-01	 6.7645814e-01	 2.0742624e-01	[ 6.2564911e-01]	 2.0222330e-01


.. parsed-literal::

      24	 6.6601615e-01	 1.9585645e-01	 7.0281329e-01	 2.0441515e-01	[ 6.4913067e-01]	 2.0642757e-01
      25	 7.0820240e-01	 1.9527583e-01	 7.4650034e-01	 2.0447228e-01	[ 6.9387153e-01]	 1.9268608e-01


.. parsed-literal::

      26	 7.3330326e-01	 1.9440955e-01	 7.7198339e-01	 2.0345088e-01	[ 7.1574930e-01]	 3.2293582e-01


.. parsed-literal::

      27	 7.5777365e-01	 1.9331156e-01	 7.9638622e-01	 2.0239229e-01	[ 7.3952376e-01]	 2.0887280e-01


.. parsed-literal::

      28	 8.0653299e-01	 1.9725149e-01	 8.4459967e-01	 2.0527984e-01	[ 7.9388895e-01]	 2.1082139e-01


.. parsed-literal::

      29	 8.4198619e-01	 1.9832691e-01	 8.8135449e-01	 2.0220070e-01	[ 8.2911730e-01]	 2.1261764e-01


.. parsed-literal::

      30	 8.7424088e-01	 1.9289054e-01	 9.1418576e-01	 1.9802493e-01	[ 8.5878888e-01]	 2.1829081e-01


.. parsed-literal::

      31	 9.0057053e-01	 1.8988689e-01	 9.4147272e-01	 1.9616293e-01	[ 8.8105350e-01]	 2.1683621e-01


.. parsed-literal::

      32	 9.3276938e-01	 1.8385036e-01	 9.7543437e-01	 1.9229360e-01	[ 9.0868267e-01]	 2.0503020e-01


.. parsed-literal::

      33	 9.5586907e-01	 1.7973326e-01	 9.9915831e-01	 1.8744797e-01	[ 9.3768176e-01]	 2.1664286e-01


.. parsed-literal::

      34	 9.7652352e-01	 1.7555364e-01	 1.0202945e+00	 1.8453492e-01	[ 9.5988524e-01]	 2.2274303e-01


.. parsed-literal::

      35	 9.9831437e-01	 1.7217165e-01	 1.0430832e+00	 1.8186800e-01	[ 9.8749720e-01]	 2.1797752e-01


.. parsed-literal::

      36	 1.0122230e+00	 1.7132880e-01	 1.0577329e+00	 1.7747653e-01	[ 1.0006839e+00]	 2.0114470e-01
      37	 1.0293284e+00	 1.6918911e-01	 1.0751366e+00	 1.7662361e-01	[ 1.0203842e+00]	 1.8578482e-01


.. parsed-literal::

      38	 1.0382998e+00	 1.6696036e-01	 1.0844981e+00	 1.7428021e-01	[ 1.0275968e+00]	 1.9820476e-01


.. parsed-literal::

      39	 1.0497509e+00	 1.6465330e-01	 1.0964994e+00	 1.7211107e-01	[ 1.0322514e+00]	 2.0752358e-01
      40	 1.0600460e+00	 1.6208497e-01	 1.1071484e+00	 1.6892381e-01	[ 1.0334968e+00]	 1.9358945e-01


.. parsed-literal::

      41	 1.0711915e+00	 1.6141499e-01	 1.1178023e+00	 1.6948068e-01	[ 1.0420364e+00]	 2.1877742e-01
      42	 1.0792167e+00	 1.6072761e-01	 1.1257037e+00	 1.6947352e-01	[ 1.0484493e+00]	 1.7850065e-01


.. parsed-literal::

      43	 1.0889056e+00	 1.6053373e-01	 1.1355270e+00	 1.6929905e-01	[ 1.0582676e+00]	 2.0048523e-01


.. parsed-literal::

      44	 1.0972418e+00	 1.5979500e-01	 1.1444992e+00	 1.6830926e-01	[ 1.0664789e+00]	 2.1317196e-01


.. parsed-literal::

      45	 1.1074726e+00	 1.5882619e-01	 1.1548081e+00	 1.6664227e-01	[ 1.0773006e+00]	 2.1485281e-01
      46	 1.1175741e+00	 1.5742645e-01	 1.1651261e+00	 1.6433988e-01	[ 1.0868523e+00]	 1.7662096e-01


.. parsed-literal::

      47	 1.1282445e+00	 1.5648155e-01	 1.1760787e+00	 1.6235747e-01	[ 1.0994900e+00]	 2.1443987e-01


.. parsed-literal::

      48	 1.1413547e+00	 1.5480132e-01	 1.1894176e+00	 1.5941505e-01	[ 1.1114743e+00]	 2.0240641e-01


.. parsed-literal::

      49	 1.1512237e+00	 1.5385427e-01	 1.1992772e+00	 1.5837542e-01	[ 1.1195475e+00]	 2.1034217e-01


.. parsed-literal::

      50	 1.1658682e+00	 1.5193143e-01	 1.2139781e+00	 1.5609181e-01	[ 1.1262466e+00]	 2.0284843e-01


.. parsed-literal::

      51	 1.1781723e+00	 1.4974032e-01	 1.2262910e+00	 1.5368916e-01	[ 1.1308848e+00]	 2.1512079e-01


.. parsed-literal::

      52	 1.1905438e+00	 1.4764045e-01	 1.2388154e+00	 1.5144485e-01	[ 1.1361651e+00]	 2.0606875e-01


.. parsed-literal::

      53	 1.2022623e+00	 1.4568290e-01	 1.2507685e+00	 1.4961034e-01	[ 1.1433320e+00]	 2.0415545e-01


.. parsed-literal::

      54	 1.2133911e+00	 1.4386435e-01	 1.2622608e+00	 1.4803874e-01	[ 1.1525144e+00]	 2.1803522e-01


.. parsed-literal::

      55	 1.2247941e+00	 1.4283632e-01	 1.2740371e+00	 1.4673342e-01	[ 1.1694716e+00]	 2.0400524e-01
      56	 1.2339272e+00	 1.4165173e-01	 1.2832153e+00	 1.4524678e-01	[ 1.1779423e+00]	 1.7443204e-01


.. parsed-literal::

      57	 1.2423245e+00	 1.4113491e-01	 1.2916470e+00	 1.4410134e-01	[ 1.1860049e+00]	 2.0224476e-01


.. parsed-literal::

      58	 1.2537499e+00	 1.4043786e-01	 1.3035599e+00	 1.4404411e-01	  1.1847801e+00 	 2.1187901e-01
      59	 1.2621721e+00	 1.4041009e-01	 1.3123728e+00	 1.4311723e-01	  1.1852319e+00 	 1.9284654e-01


.. parsed-literal::

      60	 1.2696542e+00	 1.3964153e-01	 1.3197151e+00	 1.4388247e-01	[ 1.1886385e+00]	 2.0836902e-01


.. parsed-literal::

      61	 1.2758966e+00	 1.3915176e-01	 1.3259859e+00	 1.4478440e-01	[ 1.1906873e+00]	 2.0341873e-01
      62	 1.2831984e+00	 1.3899555e-01	 1.3334944e+00	 1.4532853e-01	[ 1.1913457e+00]	 1.8686604e-01


.. parsed-literal::

      63	 1.2910591e+00	 1.3885509e-01	 1.3417463e+00	 1.4590214e-01	[ 1.1965182e+00]	 2.0972300e-01
      64	 1.2998883e+00	 1.3872891e-01	 1.3505764e+00	 1.4498156e-01	[ 1.2009793e+00]	 1.9709802e-01


.. parsed-literal::

      65	 1.3057075e+00	 1.3852812e-01	 1.3564665e+00	 1.4388440e-01	[ 1.2065531e+00]	 2.1034122e-01
      66	 1.3129579e+00	 1.3808037e-01	 1.3638231e+00	 1.4280387e-01	[ 1.2161184e+00]	 1.9912004e-01


.. parsed-literal::

      67	 1.3204851e+00	 1.3731582e-01	 1.3716862e+00	 1.4273708e-01	[ 1.2288438e+00]	 2.0636463e-01


.. parsed-literal::

      68	 1.3302931e+00	 1.3640224e-01	 1.3812991e+00	 1.4175078e-01	[ 1.2443202e+00]	 2.0911717e-01
      69	 1.3356223e+00	 1.3595892e-01	 1.3865410e+00	 1.4200181e-01	[ 1.2500114e+00]	 1.7908001e-01


.. parsed-literal::

      70	 1.3424482e+00	 1.3551520e-01	 1.3934209e+00	 1.4191236e-01	[ 1.2554291e+00]	 1.8847227e-01
      71	 1.3511201e+00	 1.3502034e-01	 1.4022841e+00	 1.4072335e-01	[ 1.2594535e+00]	 1.9899893e-01


.. parsed-literal::

      72	 1.3588262e+00	 1.3513443e-01	 1.4099563e+00	 1.3989945e-01	[ 1.2627814e+00]	 2.0438004e-01


.. parsed-literal::

      73	 1.3642776e+00	 1.3496956e-01	 1.4154706e+00	 1.3945101e-01	  1.2597599e+00 	 2.0482588e-01


.. parsed-literal::

      74	 1.3690369e+00	 1.3513589e-01	 1.4203312e+00	 1.3954732e-01	  1.2575392e+00 	 2.0888066e-01


.. parsed-literal::

      75	 1.3751788e+00	 1.3528891e-01	 1.4266681e+00	 1.4070288e-01	  1.2575750e+00 	 2.1161199e-01


.. parsed-literal::

      76	 1.3805421e+00	 1.3586160e-01	 1.4323594e+00	 1.4103264e-01	  1.2505189e+00 	 2.1172071e-01


.. parsed-literal::

      77	 1.3858198e+00	 1.3579407e-01	 1.4376892e+00	 1.4098201e-01	  1.2626852e+00 	 2.0783567e-01
      78	 1.3897406e+00	 1.3533312e-01	 1.4415728e+00	 1.4007518e-01	[ 1.2718778e+00]	 1.9861913e-01


.. parsed-literal::

      79	 1.3953262e+00	 1.3493923e-01	 1.4474032e+00	 1.3793590e-01	[ 1.2784590e+00]	 2.0899034e-01


.. parsed-literal::

      80	 1.4007290e+00	 1.3507194e-01	 1.4531385e+00	 1.3703412e-01	[ 1.2795228e+00]	 2.1856856e-01


.. parsed-literal::

      81	 1.4057340e+00	 1.3506943e-01	 1.4580571e+00	 1.3629330e-01	[ 1.2834481e+00]	 2.1072507e-01


.. parsed-literal::

      82	 1.4088566e+00	 1.3504823e-01	 1.4611167e+00	 1.3660389e-01	  1.2829529e+00 	 2.0721412e-01


.. parsed-literal::

      83	 1.4131045e+00	 1.3497115e-01	 1.4653720e+00	 1.3694713e-01	  1.2829125e+00 	 2.0240307e-01


.. parsed-literal::

      84	 1.4169161e+00	 1.3471361e-01	 1.4692585e+00	 1.3672812e-01	[ 1.2890269e+00]	 2.1268439e-01


.. parsed-literal::

      85	 1.4215478e+00	 1.3455117e-01	 1.4738638e+00	 1.3673878e-01	[ 1.2923961e+00]	 2.1676111e-01


.. parsed-literal::

      86	 1.4242942e+00	 1.3437290e-01	 1.4767006e+00	 1.3639393e-01	[ 1.2960040e+00]	 2.1336341e-01


.. parsed-literal::

      87	 1.4270537e+00	 1.3408387e-01	 1.4795336e+00	 1.3601311e-01	[ 1.2985317e+00]	 2.0795894e-01


.. parsed-literal::

      88	 1.4318369e+00	 1.3369377e-01	 1.4844485e+00	 1.3533261e-01	[ 1.3019890e+00]	 2.1144009e-01


.. parsed-literal::

      89	 1.4345457e+00	 1.3298094e-01	 1.4872801e+00	 1.3507696e-01	  1.2968912e+00 	 2.1246767e-01
      90	 1.4379370e+00	 1.3301355e-01	 1.4904577e+00	 1.3508558e-01	[ 1.3029618e+00]	 1.6893554e-01


.. parsed-literal::

      91	 1.4399686e+00	 1.3287144e-01	 1.4924096e+00	 1.3513579e-01	[ 1.3043968e+00]	 2.1211219e-01


.. parsed-literal::

      92	 1.4432096e+00	 1.3246033e-01	 1.4956213e+00	 1.3509896e-01	[ 1.3060723e+00]	 2.1225023e-01
      93	 1.4447398e+00	 1.3162345e-01	 1.4973882e+00	 1.3468406e-01	[ 1.3077424e+00]	 1.9267416e-01


.. parsed-literal::

      94	 1.4498142e+00	 1.3143751e-01	 1.5023645e+00	 1.3465869e-01	[ 1.3088738e+00]	 1.9495487e-01


.. parsed-literal::

      95	 1.4519476e+00	 1.3123981e-01	 1.5045482e+00	 1.3443231e-01	[ 1.3097581e+00]	 2.1486783e-01
      96	 1.4544842e+00	 1.3093662e-01	 1.5071765e+00	 1.3420779e-01	[ 1.3110156e+00]	 1.8077993e-01


.. parsed-literal::

      97	 1.4581264e+00	 1.3061887e-01	 1.5109010e+00	 1.3406350e-01	[ 1.3117915e+00]	 2.0779300e-01


.. parsed-literal::

      98	 1.4604297e+00	 1.3031626e-01	 1.5132710e+00	 1.3397989e-01	[ 1.3163662e+00]	 3.1927872e-01


.. parsed-literal::

      99	 1.4632640e+00	 1.3015158e-01	 1.5161174e+00	 1.3406913e-01	[ 1.3180792e+00]	 2.1424222e-01


.. parsed-literal::

     100	 1.4652578e+00	 1.3006363e-01	 1.5180761e+00	 1.3408970e-01	[ 1.3185075e+00]	 2.0099783e-01


.. parsed-literal::

     101	 1.4678934e+00	 1.2983146e-01	 1.5207056e+00	 1.3408053e-01	[ 1.3214772e+00]	 2.1696448e-01


.. parsed-literal::

     102	 1.4709429e+00	 1.2952178e-01	 1.5238192e+00	 1.3386810e-01	[ 1.3216980e+00]	 2.1562743e-01


.. parsed-literal::

     103	 1.4736318e+00	 1.2923989e-01	 1.5265715e+00	 1.3360222e-01	[ 1.3230588e+00]	 2.1089578e-01
     104	 1.4764345e+00	 1.2904625e-01	 1.5294459e+00	 1.3340578e-01	[ 1.3251381e+00]	 1.8127084e-01


.. parsed-literal::

     105	 1.4786557e+00	 1.2883856e-01	 1.5317681e+00	 1.3333211e-01	  1.3205088e+00 	 2.2293258e-01


.. parsed-literal::

     106	 1.4808146e+00	 1.2871220e-01	 1.5339491e+00	 1.3329300e-01	  1.3215695e+00 	 2.0995259e-01


.. parsed-literal::

     107	 1.4826202e+00	 1.2859185e-01	 1.5357461e+00	 1.3345018e-01	  1.3210447e+00 	 2.0416021e-01


.. parsed-literal::

     108	 1.4853393e+00	 1.2814661e-01	 1.5384331e+00	 1.3335185e-01	  1.3213103e+00 	 2.1249843e-01
     109	 1.4871075e+00	 1.2802164e-01	 1.5402318e+00	 1.3334484e-01	  1.3200139e+00 	 1.8807745e-01


.. parsed-literal::

     110	 1.4891292e+00	 1.2786110e-01	 1.5421938e+00	 1.3324768e-01	  1.3216565e+00 	 2.1695971e-01


.. parsed-literal::

     111	 1.4908761e+00	 1.2762234e-01	 1.5439653e+00	 1.3298409e-01	  1.3213045e+00 	 2.1942115e-01


.. parsed-literal::

     112	 1.4926525e+00	 1.2742757e-01	 1.5458081e+00	 1.3295804e-01	  1.3187036e+00 	 2.1301317e-01


.. parsed-literal::

     113	 1.4943717e+00	 1.2705084e-01	 1.5476627e+00	 1.3283096e-01	  1.3146135e+00 	 2.1187973e-01


.. parsed-literal::

     114	 1.4963892e+00	 1.2710737e-01	 1.5496487e+00	 1.3305840e-01	  1.3126570e+00 	 2.1305609e-01


.. parsed-literal::

     115	 1.4974417e+00	 1.2712276e-01	 1.5506682e+00	 1.3322860e-01	  1.3127915e+00 	 2.0934272e-01


.. parsed-literal::

     116	 1.4991750e+00	 1.2707992e-01	 1.5523613e+00	 1.3336046e-01	  1.3134992e+00 	 2.0803452e-01


.. parsed-literal::

     117	 1.5001197e+00	 1.2699385e-01	 1.5533254e+00	 1.3367013e-01	  1.3104509e+00 	 2.0817184e-01


.. parsed-literal::

     118	 1.5025109e+00	 1.2690069e-01	 1.5556404e+00	 1.3350663e-01	  1.3138430e+00 	 2.2124505e-01
     119	 1.5035675e+00	 1.2680748e-01	 1.5566929e+00	 1.3337222e-01	  1.3145029e+00 	 1.9129467e-01


.. parsed-literal::

     120	 1.5047903e+00	 1.2671033e-01	 1.5579328e+00	 1.3328084e-01	  1.3143236e+00 	 2.0242691e-01


.. parsed-literal::

     121	 1.5069761e+00	 1.2656321e-01	 1.5601768e+00	 1.3322510e-01	  1.3132870e+00 	 2.0824671e-01
     122	 1.5092372e+00	 1.2648254e-01	 1.5626727e+00	 1.3375851e-01	  1.3040584e+00 	 1.7430139e-01


.. parsed-literal::

     123	 1.5117651e+00	 1.2636003e-01	 1.5652019e+00	 1.3346359e-01	  1.3034052e+00 	 2.1491265e-01


.. parsed-literal::

     124	 1.5129245e+00	 1.2634043e-01	 1.5662940e+00	 1.3351974e-01	  1.3059697e+00 	 2.0533943e-01
     125	 1.5143823e+00	 1.2626828e-01	 1.5677312e+00	 1.3355481e-01	  1.3068048e+00 	 1.9344187e-01


.. parsed-literal::

     126	 1.5157240e+00	 1.2620147e-01	 1.5691174e+00	 1.3367919e-01	  1.3037791e+00 	 2.0413351e-01
     127	 1.5172710e+00	 1.2607622e-01	 1.5706504e+00	 1.3347113e-01	  1.3053390e+00 	 1.7074585e-01


.. parsed-literal::

     128	 1.5188519e+00	 1.2590667e-01	 1.5722697e+00	 1.3313179e-01	  1.3049404e+00 	 2.0513892e-01


.. parsed-literal::

     129	 1.5201413e+00	 1.2583484e-01	 1.5735770e+00	 1.3293830e-01	  1.3038533e+00 	 2.0889330e-01
     130	 1.5228093e+00	 1.2542451e-01	 1.5764591e+00	 1.3224601e-01	  1.2987903e+00 	 1.9309521e-01


.. parsed-literal::

     131	 1.5237773e+00	 1.2553839e-01	 1.5774655e+00	 1.3190498e-01	  1.2918871e+00 	 2.0724320e-01
     132	 1.5260494e+00	 1.2559342e-01	 1.5796073e+00	 1.3227595e-01	  1.2957715e+00 	 1.7705345e-01


.. parsed-literal::

     133	 1.5266966e+00	 1.2555424e-01	 1.5802656e+00	 1.3226309e-01	  1.2951912e+00 	 2.1641088e-01
     134	 1.5281645e+00	 1.2551545e-01	 1.5817759e+00	 1.3222028e-01	  1.2919288e+00 	 1.9764733e-01


.. parsed-literal::

     135	 1.5298074e+00	 1.2537002e-01	 1.5834471e+00	 1.3209930e-01	  1.2911494e+00 	 2.1336341e-01


.. parsed-literal::

     136	 1.5307688e+00	 1.2535854e-01	 1.5845116e+00	 1.3179560e-01	  1.2829911e+00 	 2.1240997e-01


.. parsed-literal::

     137	 1.5324631e+00	 1.2521874e-01	 1.5861118e+00	 1.3178227e-01	  1.2879761e+00 	 2.1560740e-01


.. parsed-literal::

     138	 1.5332021e+00	 1.2509528e-01	 1.5868352e+00	 1.3163662e-01	  1.2895180e+00 	 2.0167279e-01


.. parsed-literal::

     139	 1.5345007e+00	 1.2490868e-01	 1.5881388e+00	 1.3137114e-01	  1.2895290e+00 	 2.1271062e-01


.. parsed-literal::

     140	 1.5355497e+00	 1.2463982e-01	 1.5893075e+00	 1.3102694e-01	  1.2842501e+00 	 2.2012734e-01
     141	 1.5372231e+00	 1.2455803e-01	 1.5909820e+00	 1.3080884e-01	  1.2829055e+00 	 1.9951701e-01


.. parsed-literal::

     142	 1.5381576e+00	 1.2456966e-01	 1.5919299e+00	 1.3070394e-01	  1.2814129e+00 	 2.0123172e-01


.. parsed-literal::

     143	 1.5392358e+00	 1.2456855e-01	 1.5930660e+00	 1.3063772e-01	  1.2771545e+00 	 2.0446754e-01


.. parsed-literal::

     144	 1.5400598e+00	 1.2436725e-01	 1.5939997e+00	 1.3017502e-01	  1.2769708e+00 	 2.0448232e-01


.. parsed-literal::

     145	 1.5415464e+00	 1.2435001e-01	 1.5954602e+00	 1.3027929e-01	  1.2753842e+00 	 2.1356249e-01


.. parsed-literal::

     146	 1.5426436e+00	 1.2422827e-01	 1.5965674e+00	 1.3026398e-01	  1.2749898e+00 	 2.1531916e-01


.. parsed-literal::

     147	 1.5433484e+00	 1.2412942e-01	 1.5972766e+00	 1.3017505e-01	  1.2754565e+00 	 2.1617270e-01


.. parsed-literal::

     148	 1.5450471e+00	 1.2392498e-01	 1.5990156e+00	 1.2990004e-01	  1.2720295e+00 	 2.0911217e-01


.. parsed-literal::

     149	 1.5457948e+00	 1.2393905e-01	 1.5998006e+00	 1.2986294e-01	  1.2705104e+00 	 3.2595778e-01


.. parsed-literal::

     150	 1.5469236e+00	 1.2385566e-01	 1.6009774e+00	 1.2961241e-01	  1.2674271e+00 	 2.0326185e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.02 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f396c35ab00>



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
    CPU times: user 2.12 s, sys: 44 ms, total: 2.16 s
    Wall time: 656 ms


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

