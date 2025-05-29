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
       1	-3.4715976e-01	 3.2178050e-01	-3.3747007e-01	 3.1498867e-01	[-3.2491130e-01]	 4.6398926e-01


.. parsed-literal::

       2	-2.7693628e-01	 3.1128861e-01	-2.5296676e-01	 3.0488063e-01	[-2.3370901e-01]	 2.3179913e-01


.. parsed-literal::

       3	-2.3380036e-01	 2.9082719e-01	-1.9166676e-01	 2.8283436e-01	[-1.6131753e-01]	 2.8686070e-01
       4	-1.9349094e-01	 2.6712792e-01	-1.5091023e-01	 2.6211699e-01	[-1.2456106e-01]	 1.7099857e-01


.. parsed-literal::

       5	-1.0480491e-01	 2.5695243e-01	-7.0795421e-02	 2.5277054e-01	[-5.2107834e-02]	 2.1602821e-01


.. parsed-literal::

       6	-7.4947244e-02	 2.5303480e-01	-4.4988164e-02	 2.4815423e-01	[-2.7639331e-02]	 2.2091675e-01


.. parsed-literal::

       7	-5.6412833e-02	 2.4982771e-01	-3.2288380e-02	 2.4530364e-01	[-1.4601820e-02]	 2.1891451e-01
       8	-4.4801800e-02	 2.4796054e-01	-2.4486792e-02	 2.4267555e-01	[-3.2839178e-03]	 2.0962143e-01


.. parsed-literal::

       9	-3.2714694e-02	 2.4576172e-01	-1.5128121e-02	 2.3959512e-01	[ 9.3877080e-03]	 2.1056271e-01
      10	-2.1223764e-02	 2.4307635e-01	-5.6366848e-03	 2.3822730e-01	[ 1.5931957e-02]	 1.9532561e-01


.. parsed-literal::

      11	-1.8705404e-02	 2.4251082e-01	-4.8295349e-03	 2.3926740e-01	  6.9120245e-03 	 1.8327641e-01


.. parsed-literal::

      12	-1.1830670e-02	 2.4189689e-01	 2.1046584e-03	 2.3839284e-01	[ 1.6225925e-02]	 2.1151686e-01
      13	-9.1670401e-03	 2.4128536e-01	 4.9772502e-03	 2.3760924e-01	[ 2.0471953e-02]	 1.9225788e-01


.. parsed-literal::

      14	-5.8474311e-03	 2.4044629e-01	 8.9262601e-03	 2.3681046e-01	[ 2.4551195e-02]	 1.8974876e-01


.. parsed-literal::

      15	 2.0740969e-01	 2.2487412e-01	 2.3356333e-01	 2.2535125e-01	[ 2.2664312e-01]	 4.3256497e-01


.. parsed-literal::

      16	 2.8388052e-01	 2.2021909e-01	 3.1139900e-01	 2.1951225e-01	[ 3.0736930e-01]	 2.2034144e-01
      17	 3.8952043e-01	 2.1521503e-01	 4.2162753e-01	 2.1835851e-01	[ 4.0762613e-01]	 1.8742895e-01


.. parsed-literal::

      18	 4.6860425e-01	 2.1286427e-01	 5.0475070e-01	 2.1514128e-01	[ 4.9292889e-01]	 1.7423606e-01


.. parsed-literal::

      19	 5.1722693e-01	 2.1155266e-01	 5.5382069e-01	 2.1286396e-01	[ 5.3709384e-01]	 2.1418166e-01


.. parsed-literal::

      20	 5.5823459e-01	 2.0759265e-01	 5.9534793e-01	 2.0864344e-01	[ 5.7371374e-01]	 2.0484996e-01


.. parsed-literal::

      21	 6.2756228e-01	 2.0019997e-01	 6.6747566e-01	 2.0054713e-01	[ 6.3435288e-01]	 2.1884155e-01
      22	 6.7086445e-01	 1.9950114e-01	 7.0979793e-01	 1.9916166e-01	[ 6.6858378e-01]	 1.9666553e-01


.. parsed-literal::

      23	 7.0593613e-01	 1.9630005e-01	 7.4445455e-01	 1.9700352e-01	[ 6.9944712e-01]	 1.8086171e-01


.. parsed-literal::

      24	 7.2590998e-01	 2.0313005e-01	 7.6405740e-01	 2.0069469e-01	[ 7.2697368e-01]	 2.1864319e-01


.. parsed-literal::

      25	 7.8481286e-01	 1.9117613e-01	 8.2363030e-01	 1.9041754e-01	[ 7.7774001e-01]	 2.1270370e-01


.. parsed-literal::

      26	 8.1063781e-01	 1.8932320e-01	 8.5028928e-01	 1.8979370e-01	[ 7.9679660e-01]	 2.0331430e-01


.. parsed-literal::

      27	 8.3866049e-01	 1.8869987e-01	 8.7984936e-01	 1.8818936e-01	[ 8.1941039e-01]	 2.0797086e-01


.. parsed-literal::

      28	 8.6260633e-01	 1.9039752e-01	 9.0392692e-01	 1.9072461e-01	[ 8.5452973e-01]	 2.1328211e-01


.. parsed-literal::

      29	 8.8919896e-01	 1.9093128e-01	 9.3194387e-01	 1.8724268e-01	[ 8.8720105e-01]	 2.0858598e-01


.. parsed-literal::

      30	 9.1509098e-01	 1.8741348e-01	 9.5795409e-01	 1.8425354e-01	[ 9.0898247e-01]	 2.1435428e-01
      31	 9.3663648e-01	 1.8423464e-01	 9.7955926e-01	 1.8090274e-01	[ 9.3447726e-01]	 1.9289160e-01


.. parsed-literal::

      32	 9.5630680e-01	 1.8087538e-01	 1.0011226e+00	 1.7769129e-01	[ 9.5827469e-01]	 1.8935466e-01


.. parsed-literal::

      33	 9.7628195e-01	 1.7786971e-01	 1.0207098e+00	 1.7576183e-01	[ 9.7970068e-01]	 2.0856309e-01


.. parsed-literal::

      34	 9.8776526e-01	 1.7581258e-01	 1.0321039e+00	 1.7457041e-01	[ 9.8738270e-01]	 2.0222163e-01
      35	 1.0064198e+00	 1.7394383e-01	 1.0518470e+00	 1.7335219e-01	[ 1.0047051e+00]	 1.7572308e-01


.. parsed-literal::

      36	 1.0157098e+00	 1.7301344e-01	 1.0619769e+00	 1.7197379e-01	[ 1.0123709e+00]	 1.8458939e-01
      37	 1.0256256e+00	 1.7236805e-01	 1.0719343e+00	 1.7110273e-01	[ 1.0211901e+00]	 1.7552018e-01


.. parsed-literal::

      38	 1.0348927e+00	 1.7153113e-01	 1.0816316e+00	 1.6995733e-01	[ 1.0276230e+00]	 2.1318340e-01


.. parsed-literal::

      39	 1.0442379e+00	 1.7074336e-01	 1.0910226e+00	 1.6866713e-01	[ 1.0326629e+00]	 2.0816183e-01


.. parsed-literal::

      40	 1.0626540e+00	 1.6989115e-01	 1.1099138e+00	 1.6618762e-01	[ 1.0514548e+00]	 2.1776652e-01


.. parsed-literal::

      41	 1.0680420e+00	 1.6759404e-01	 1.1155522e+00	 1.6346743e-01	  1.0466137e+00 	 2.1622729e-01


.. parsed-literal::

      42	 1.0795027e+00	 1.6741637e-01	 1.1261906e+00	 1.6289165e-01	[ 1.0654684e+00]	 2.1041536e-01


.. parsed-literal::

      43	 1.0864171e+00	 1.6663951e-01	 1.1331898e+00	 1.6224213e-01	[ 1.0729657e+00]	 2.1219254e-01


.. parsed-literal::

      44	 1.0957560e+00	 1.6501492e-01	 1.1426945e+00	 1.6060537e-01	[ 1.0835412e+00]	 2.0746446e-01


.. parsed-literal::

      45	 1.1123292e+00	 1.6171775e-01	 1.1591855e+00	 1.5760756e-01	[ 1.0998003e+00]	 2.0916533e-01


.. parsed-literal::

      46	 1.1172932e+00	 1.6146138e-01	 1.1656218e+00	 1.5532783e-01	[ 1.1007238e+00]	 2.1020865e-01


.. parsed-literal::

      47	 1.1329889e+00	 1.5934430e-01	 1.1805727e+00	 1.5384955e-01	[ 1.1166789e+00]	 2.0568132e-01


.. parsed-literal::

      48	 1.1413875e+00	 1.5801391e-01	 1.1889585e+00	 1.5299086e-01	[ 1.1240692e+00]	 2.1289635e-01


.. parsed-literal::

      49	 1.1528141e+00	 1.5585629e-01	 1.2008000e+00	 1.5058373e-01	[ 1.1318804e+00]	 2.1320653e-01


.. parsed-literal::

      50	 1.1620477e+00	 1.5408133e-01	 1.2108601e+00	 1.5024832e-01	[ 1.1384037e+00]	 2.1163297e-01
      51	 1.1718599e+00	 1.5273898e-01	 1.2207660e+00	 1.4853209e-01	[ 1.1478493e+00]	 1.9880080e-01


.. parsed-literal::

      52	 1.1809870e+00	 1.5222681e-01	 1.2301568e+00	 1.4735736e-01	[ 1.1561933e+00]	 1.8914008e-01


.. parsed-literal::

      53	 1.1890715e+00	 1.5131931e-01	 1.2385406e+00	 1.4640517e-01	[ 1.1704405e+00]	 2.1150494e-01


.. parsed-literal::

      54	 1.2017367e+00	 1.5006264e-01	 1.2517014e+00	 1.4540341e-01	[ 1.1887670e+00]	 2.1217251e-01


.. parsed-literal::

      55	 1.2134445e+00	 1.4904662e-01	 1.2635668e+00	 1.4428418e-01	[ 1.2008744e+00]	 2.1204138e-01


.. parsed-literal::

      56	 1.2235918e+00	 1.4717604e-01	 1.2738472e+00	 1.4253791e-01	[ 1.2191260e+00]	 2.0623374e-01
      57	 1.2329243e+00	 1.4616462e-01	 1.2831122e+00	 1.4186790e-01	[ 1.2243493e+00]	 1.9401264e-01


.. parsed-literal::

      58	 1.2434637e+00	 1.4480572e-01	 1.2937670e+00	 1.4113259e-01	[ 1.2278445e+00]	 2.0715833e-01


.. parsed-literal::

      59	 1.2496524e+00	 1.4352618e-01	 1.3002323e+00	 1.4074880e-01	  1.2278085e+00 	 2.0411205e-01
      60	 1.2572156e+00	 1.4303327e-01	 1.3076654e+00	 1.4069628e-01	[ 1.2315548e+00]	 1.8814611e-01


.. parsed-literal::

      61	 1.2632411e+00	 1.4252368e-01	 1.3136074e+00	 1.4069548e-01	[ 1.2362394e+00]	 1.9950175e-01


.. parsed-literal::

      62	 1.2693805e+00	 1.4204443e-01	 1.3198565e+00	 1.4068851e-01	[ 1.2383436e+00]	 2.1106029e-01


.. parsed-literal::

      63	 1.2745169e+00	 1.4154432e-01	 1.3249377e+00	 1.4110704e-01	[ 1.2412405e+00]	 3.4094954e-01
      64	 1.2819247e+00	 1.4128880e-01	 1.3324999e+00	 1.4138760e-01	  1.2408933e+00 	 1.7661738e-01


.. parsed-literal::

      65	 1.2872683e+00	 1.4135352e-01	 1.3378942e+00	 1.4158413e-01	  1.2397455e+00 	 2.0763683e-01


.. parsed-literal::

      66	 1.2947134e+00	 1.4113309e-01	 1.3456056e+00	 1.4212366e-01	  1.2354595e+00 	 2.1294522e-01


.. parsed-literal::

      67	 1.3008963e+00	 1.4084132e-01	 1.3519334e+00	 1.4189878e-01	  1.2350851e+00 	 2.1767521e-01


.. parsed-literal::

      68	 1.3072193e+00	 1.4018489e-01	 1.3583067e+00	 1.4150303e-01	[ 1.2420727e+00]	 2.0931721e-01


.. parsed-literal::

      69	 1.3179246e+00	 1.3901241e-01	 1.3696347e+00	 1.4065661e-01	[ 1.2492197e+00]	 2.1226430e-01
      70	 1.3222364e+00	 1.3887796e-01	 1.3741128e+00	 1.4089702e-01	[ 1.2573137e+00]	 1.8581414e-01


.. parsed-literal::

      71	 1.3284908e+00	 1.3855179e-01	 1.3801649e+00	 1.4030573e-01	[ 1.2642943e+00]	 2.0669222e-01


.. parsed-literal::

      72	 1.3336396e+00	 1.3829432e-01	 1.3853974e+00	 1.3992238e-01	[ 1.2667235e+00]	 2.0969677e-01


.. parsed-literal::

      73	 1.3383581e+00	 1.3805982e-01	 1.3902173e+00	 1.3941615e-01	[ 1.2688059e+00]	 2.2143722e-01


.. parsed-literal::

      74	 1.3434477e+00	 1.3741155e-01	 1.3956881e+00	 1.3885383e-01	[ 1.2744475e+00]	 2.1866441e-01


.. parsed-literal::

      75	 1.3491659e+00	 1.3714732e-01	 1.4013508e+00	 1.3858267e-01	  1.2735266e+00 	 2.0925140e-01


.. parsed-literal::

      76	 1.3532152e+00	 1.3688247e-01	 1.4054527e+00	 1.3854863e-01	[ 1.2747256e+00]	 2.1509504e-01


.. parsed-literal::

      77	 1.3598301e+00	 1.3632212e-01	 1.4121216e+00	 1.3854668e-01	[ 1.2786913e+00]	 2.1057487e-01
      78	 1.3638399e+00	 1.3595265e-01	 1.4164087e+00	 1.3883257e-01	  1.2759799e+00 	 1.9027066e-01


.. parsed-literal::

      79	 1.3699048e+00	 1.3555341e-01	 1.4222400e+00	 1.3833086e-01	[ 1.2834615e+00]	 3.7593031e-01


.. parsed-literal::

      80	 1.3728969e+00	 1.3544267e-01	 1.4252373e+00	 1.3815000e-01	[ 1.2874950e+00]	 2.1061969e-01


.. parsed-literal::

      81	 1.3774694e+00	 1.3525376e-01	 1.4299087e+00	 1.3787150e-01	[ 1.2930456e+00]	 2.1418071e-01


.. parsed-literal::

      82	 1.3826029e+00	 1.3501353e-01	 1.4353363e+00	 1.3734045e-01	[ 1.3013656e+00]	 2.0780778e-01
      83	 1.3894470e+00	 1.3458676e-01	 1.4422122e+00	 1.3725133e-01	[ 1.3083612e+00]	 1.8021226e-01


.. parsed-literal::

      84	 1.3955134e+00	 1.3412627e-01	 1.4484451e+00	 1.3716278e-01	[ 1.3095204e+00]	 2.0732665e-01


.. parsed-literal::

      85	 1.3998840e+00	 1.3389656e-01	 1.4529853e+00	 1.3735875e-01	  1.3075949e+00 	 2.0893049e-01


.. parsed-literal::

      86	 1.4048664e+00	 1.3349426e-01	 1.4580226e+00	 1.3749933e-01	  1.3089844e+00 	 2.1015239e-01


.. parsed-literal::

      87	 1.4109678e+00	 1.3332431e-01	 1.4642149e+00	 1.3735408e-01	[ 1.3101076e+00]	 2.0955253e-01


.. parsed-literal::

      88	 1.4148321e+00	 1.3286768e-01	 1.4683012e+00	 1.3720575e-01	[ 1.3114541e+00]	 2.1864891e-01


.. parsed-literal::

      89	 1.4188820e+00	 1.3289464e-01	 1.4722567e+00	 1.3688921e-01	[ 1.3147738e+00]	 2.0663714e-01
      90	 1.4229576e+00	 1.3288770e-01	 1.4764414e+00	 1.3682526e-01	[ 1.3172277e+00]	 1.8647003e-01


.. parsed-literal::

      91	 1.4270298e+00	 1.3273343e-01	 1.4806269e+00	 1.3702644e-01	[ 1.3172967e+00]	 1.7381310e-01
      92	 1.4316836e+00	 1.3276291e-01	 1.4854406e+00	 1.3771990e-01	[ 1.3180798e+00]	 1.9614100e-01


.. parsed-literal::

      93	 1.4355243e+00	 1.3254868e-01	 1.4892554e+00	 1.3786223e-01	  1.3176975e+00 	 2.1365237e-01


.. parsed-literal::

      94	 1.4386130e+00	 1.3243481e-01	 1.4922601e+00	 1.3791264e-01	[ 1.3207422e+00]	 2.0903897e-01


.. parsed-literal::

      95	 1.4427894e+00	 1.3229887e-01	 1.4965321e+00	 1.3791300e-01	[ 1.3220535e+00]	 2.0384741e-01


.. parsed-literal::

      96	 1.4474182e+00	 1.3226127e-01	 1.5011776e+00	 1.3777603e-01	[ 1.3258629e+00]	 2.1507287e-01


.. parsed-literal::

      97	 1.4508466e+00	 1.3206058e-01	 1.5045968e+00	 1.3782148e-01	[ 1.3276207e+00]	 2.0613360e-01


.. parsed-literal::

      98	 1.4539602e+00	 1.3178420e-01	 1.5077405e+00	 1.3776726e-01	[ 1.3279765e+00]	 2.0876956e-01
      99	 1.4565364e+00	 1.3157413e-01	 1.5104048e+00	 1.3788625e-01	  1.3276693e+00 	 1.8608928e-01


.. parsed-literal::

     100	 1.4594615e+00	 1.3126655e-01	 1.5133320e+00	 1.3765920e-01	[ 1.3287760e+00]	 2.0478415e-01


.. parsed-literal::

     101	 1.4631847e+00	 1.3096541e-01	 1.5171585e+00	 1.3753416e-01	[ 1.3289096e+00]	 2.0929003e-01
     102	 1.4656450e+00	 1.3074918e-01	 1.5196608e+00	 1.3717149e-01	  1.3279869e+00 	 1.8586278e-01


.. parsed-literal::

     103	 1.4688687e+00	 1.3064593e-01	 1.5228968e+00	 1.3719540e-01	[ 1.3295659e+00]	 2.1349573e-01
     104	 1.4723296e+00	 1.3045973e-01	 1.5263907e+00	 1.3732427e-01	  1.3287878e+00 	 1.7789793e-01


.. parsed-literal::

     105	 1.4754088e+00	 1.3029365e-01	 1.5295252e+00	 1.3696931e-01	  1.3282195e+00 	 2.1091342e-01


.. parsed-literal::

     106	 1.4779246e+00	 1.3010063e-01	 1.5321088e+00	 1.3700755e-01	  1.3249225e+00 	 2.1114564e-01
     107	 1.4803387e+00	 1.3005050e-01	 1.5346713e+00	 1.3691004e-01	  1.3229890e+00 	 1.9161153e-01


.. parsed-literal::

     108	 1.4822512e+00	 1.2991413e-01	 1.5366867e+00	 1.3666723e-01	  1.3183420e+00 	 2.0543694e-01


.. parsed-literal::

     109	 1.4844691e+00	 1.2984825e-01	 1.5389888e+00	 1.3649802e-01	  1.3183802e+00 	 2.0863700e-01
     110	 1.4880540e+00	 1.2951876e-01	 1.5427469e+00	 1.3585532e-01	  1.3172326e+00 	 2.0429111e-01


.. parsed-literal::

     111	 1.4895504e+00	 1.2936632e-01	 1.5443976e+00	 1.3558486e-01	  1.3257282e+00 	 1.8360996e-01


.. parsed-literal::

     112	 1.4916533e+00	 1.2922690e-01	 1.5463486e+00	 1.3549562e-01	  1.3267295e+00 	 2.1209049e-01


.. parsed-literal::

     113	 1.4932306e+00	 1.2906048e-01	 1.5479164e+00	 1.3533155e-01	  1.3260634e+00 	 2.1477294e-01
     114	 1.4951082e+00	 1.2890735e-01	 1.5498020e+00	 1.3514634e-01	  1.3271115e+00 	 2.0320654e-01


.. parsed-literal::

     115	 1.4980658e+00	 1.2880855e-01	 1.5528972e+00	 1.3507057e-01	  1.3254276e+00 	 2.0313978e-01


.. parsed-literal::

     116	 1.4995647e+00	 1.2861036e-01	 1.5545396e+00	 1.3477234e-01	  1.3245982e+00 	 2.1055865e-01


.. parsed-literal::

     117	 1.5017795e+00	 1.2857757e-01	 1.5566922e+00	 1.3485560e-01	  1.3249298e+00 	 2.0369649e-01


.. parsed-literal::

     118	 1.5029617e+00	 1.2856247e-01	 1.5578295e+00	 1.3487241e-01	  1.3269690e+00 	 2.1228004e-01


.. parsed-literal::

     119	 1.5050532e+00	 1.2838238e-01	 1.5598729e+00	 1.3467125e-01	  1.3283565e+00 	 2.0294929e-01


.. parsed-literal::

     120	 1.5068604e+00	 1.2815775e-01	 1.5616570e+00	 1.3426206e-01	[ 1.3326892e+00]	 2.0426130e-01


.. parsed-literal::

     121	 1.5088160e+00	 1.2794920e-01	 1.5634990e+00	 1.3399902e-01	[ 1.3338075e+00]	 2.0855117e-01


.. parsed-literal::

     122	 1.5097828e+00	 1.2782409e-01	 1.5644607e+00	 1.3382052e-01	  1.3332674e+00 	 2.0400095e-01
     123	 1.5111415e+00	 1.2768516e-01	 1.5658262e+00	 1.3358626e-01	  1.3333594e+00 	 1.9847488e-01


.. parsed-literal::

     124	 1.5139409e+00	 1.2746213e-01	 1.5687212e+00	 1.3321852e-01	  1.3312543e+00 	 2.0107841e-01


.. parsed-literal::

     125	 1.5154573e+00	 1.2727832e-01	 1.5703688e+00	 1.3287036e-01	  1.3270398e+00 	 3.2343459e-01


.. parsed-literal::

     126	 1.5173467e+00	 1.2729062e-01	 1.5723442e+00	 1.3279391e-01	  1.3263227e+00 	 2.1089554e-01


.. parsed-literal::

     127	 1.5185549e+00	 1.2725910e-01	 1.5736005e+00	 1.3275937e-01	  1.3240156e+00 	 2.1267605e-01


.. parsed-literal::

     128	 1.5199409e+00	 1.2733599e-01	 1.5750349e+00	 1.3274812e-01	  1.3233156e+00 	 2.1487308e-01


.. parsed-literal::

     129	 1.5216703e+00	 1.2730620e-01	 1.5767770e+00	 1.3271644e-01	  1.3198388e+00 	 2.0632648e-01


.. parsed-literal::

     130	 1.5234543e+00	 1.2723007e-01	 1.5785416e+00	 1.3258381e-01	  1.3190667e+00 	 2.0905209e-01


.. parsed-literal::

     131	 1.5251055e+00	 1.2716313e-01	 1.5801837e+00	 1.3239693e-01	  1.3187524e+00 	 2.0922351e-01


.. parsed-literal::

     132	 1.5266721e+00	 1.2707383e-01	 1.5817622e+00	 1.3229639e-01	  1.3183923e+00 	 2.1057606e-01


.. parsed-literal::

     133	 1.5280962e+00	 1.2686423e-01	 1.5832063e+00	 1.3208910e-01	  1.3167200e+00 	 2.1456480e-01


.. parsed-literal::

     134	 1.5298179e+00	 1.2670760e-01	 1.5850067e+00	 1.3200993e-01	  1.3126315e+00 	 2.0846152e-01
     135	 1.5313073e+00	 1.2652242e-01	 1.5865365e+00	 1.3194777e-01	  1.3092161e+00 	 1.9379067e-01


.. parsed-literal::

     136	 1.5328487e+00	 1.2635688e-01	 1.5880947e+00	 1.3191217e-01	  1.3056858e+00 	 2.0569658e-01


.. parsed-literal::

     137	 1.5345816e+00	 1.2612408e-01	 1.5898389e+00	 1.3194507e-01	  1.3036133e+00 	 2.0774698e-01


.. parsed-literal::

     138	 1.5361059e+00	 1.2576112e-01	 1.5913589e+00	 1.3171509e-01	  1.2999525e+00 	 2.1188140e-01


.. parsed-literal::

     139	 1.5372908e+00	 1.2566212e-01	 1.5925007e+00	 1.3175364e-01	  1.3003423e+00 	 2.1155047e-01


.. parsed-literal::

     140	 1.5384505e+00	 1.2548753e-01	 1.5936251e+00	 1.3168261e-01	  1.3010302e+00 	 2.0182681e-01
     141	 1.5399251e+00	 1.2528776e-01	 1.5951411e+00	 1.3164721e-01	  1.2975383e+00 	 2.0619488e-01


.. parsed-literal::

     142	 1.5404888e+00	 1.2483389e-01	 1.5959994e+00	 1.3142044e-01	  1.2895616e+00 	 2.1575665e-01
     143	 1.5423844e+00	 1.2487276e-01	 1.5977885e+00	 1.3143127e-01	  1.2897860e+00 	 1.9116545e-01


.. parsed-literal::

     144	 1.5428593e+00	 1.2486081e-01	 1.5982675e+00	 1.3141028e-01	  1.2896871e+00 	 2.1350861e-01
     145	 1.5441903e+00	 1.2460619e-01	 1.5996636e+00	 1.3110883e-01	  1.2884361e+00 	 1.8220615e-01


.. parsed-literal::

     146	 1.5451657e+00	 1.2440128e-01	 1.6007300e+00	 1.3082983e-01	  1.2882050e+00 	 2.1816325e-01


.. parsed-literal::

     147	 1.5459483e+00	 1.2421068e-01	 1.6014982e+00	 1.3069035e-01	  1.2882596e+00 	 2.1147037e-01


.. parsed-literal::

     148	 1.5469659e+00	 1.2391539e-01	 1.6024956e+00	 1.3045526e-01	  1.2877308e+00 	 2.1812725e-01
     149	 1.5480688e+00	 1.2365275e-01	 1.6035807e+00	 1.3026196e-01	  1.2858906e+00 	 1.8922210e-01


.. parsed-literal::

     150	 1.5499505e+00	 1.2318028e-01	 1.6054924e+00	 1.3003619e-01	  1.2773461e+00 	 2.1389318e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.12 s, total: 2min 6s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fce64b70e20>



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
    CPU times: user 1.75 s, sys: 42.9 ms, total: 1.79 s
    Wall time: 557 ms


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

