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
       1	-3.4855048e-01	 3.2189739e-01	-3.3895148e-01	 3.1555409e-01	[-3.2768367e-01]	 4.6196365e-01


.. parsed-literal::

       2	-2.7657991e-01	 3.1115429e-01	-2.5262405e-01	 3.0581585e-01	[-2.3706435e-01]	 2.3078585e-01


.. parsed-literal::

       3	-2.3133792e-01	 2.8986993e-01	-1.8858645e-01	 2.8666508e-01	[-1.7633022e-01]	 2.9401922e-01


.. parsed-literal::

       4	-2.0093472e-01	 2.6659693e-01	-1.6063081e-01	 2.6439576e-01	[-1.3974491e-01]	 2.1371698e-01


.. parsed-literal::

       5	-1.0969022e-01	 2.5885966e-01	-7.5405579e-02	 2.5425942e-01	[-5.3699564e-02]	 2.1092772e-01


.. parsed-literal::

       6	-7.4695270e-02	 2.5255111e-01	-4.4141461e-02	 2.4729370e-01	[-2.3711258e-02]	 2.2018147e-01


.. parsed-literal::

       7	-5.6879992e-02	 2.5002499e-01	-3.3211331e-02	 2.4409720e-01	[-1.0249655e-02]	 2.0215631e-01


.. parsed-literal::

       8	-4.2336860e-02	 2.4742609e-01	-2.2434532e-02	 2.4168081e-01	[ 2.9818615e-04]	 2.1237040e-01
       9	-2.6475587e-02	 2.4436510e-01	-9.6173609e-03	 2.4014468e-01	[ 9.4573023e-03]	 2.0220947e-01


.. parsed-literal::

      10	-2.1440307e-02	 2.4389398e-01	-6.5901914e-03	 2.4028677e-01	[ 1.0681246e-02]	 2.0515585e-01


.. parsed-literal::

      11	-1.4790599e-02	 2.4226453e-01	-2.1869777e-04	 2.3941734e-01	[ 1.4082196e-02]	 2.1163678e-01


.. parsed-literal::

      12	-1.1859949e-02	 2.4166186e-01	 2.5215810e-03	 2.3891345e-01	[ 1.6199181e-02]	 2.1973515e-01


.. parsed-literal::

      13	-8.4239876e-03	 2.4097907e-01	 5.8896185e-03	 2.3812216e-01	[ 1.9363804e-02]	 2.1783257e-01


.. parsed-literal::

      14	-4.2557653e-04	 2.3945471e-01	 1.4291153e-02	 2.3590433e-01	[ 3.0331397e-02]	 2.1899772e-01


.. parsed-literal::

      15	 1.2493669e-01	 2.2373865e-01	 1.4666795e-01	 2.1988681e-01	[ 1.5713151e-01]	 3.3114672e-01


.. parsed-literal::

      16	 1.8334378e-01	 2.2361613e-01	 2.0842480e-01	 2.1757659e-01	[ 2.2819261e-01]	 2.1511459e-01


.. parsed-literal::

      17	 2.6190580e-01	 2.2175759e-01	 2.9132325e-01	 2.1696175e-01	[ 3.1296537e-01]	 2.1239710e-01


.. parsed-literal::

      18	 3.3474410e-01	 2.1245615e-01	 3.6787604e-01	 2.1113255e-01	[ 3.8463258e-01]	 2.0133877e-01


.. parsed-literal::

      19	 3.7230370e-01	 2.1089458e-01	 4.0612612e-01	 2.1066967e-01	[ 4.1028258e-01]	 2.0475292e-01


.. parsed-literal::

      20	 4.1451177e-01	 2.0703375e-01	 4.4859899e-01	 2.0743011e-01	[ 4.5256555e-01]	 2.0626831e-01
      21	 4.8713277e-01	 2.0317190e-01	 5.2235804e-01	 2.0345154e-01	[ 5.2305032e-01]	 1.9827414e-01


.. parsed-literal::

      22	 5.7557174e-01	 2.0206850e-01	 6.1337273e-01	 2.0289251e-01	[ 6.2007659e-01]	 1.8411779e-01


.. parsed-literal::

      23	 6.0738770e-01	 1.9892465e-01	 6.4797458e-01	 1.9744695e-01	[ 6.6419958e-01]	 2.1394348e-01


.. parsed-literal::

      24	 6.4645661e-01	 1.9764268e-01	 6.8431574e-01	 1.9592623e-01	[ 7.0449786e-01]	 2.1678972e-01


.. parsed-literal::

      25	 6.7492128e-01	 1.9850016e-01	 7.1268739e-01	 1.9583978e-01	[ 7.3602544e-01]	 2.0824695e-01


.. parsed-literal::

      26	 7.1002069e-01	 2.0682221e-01	 7.4626713e-01	 2.0432480e-01	[ 7.6496583e-01]	 2.0487905e-01


.. parsed-literal::

      27	 7.7740725e-01	 2.0109621e-01	 8.1612709e-01	 1.9976253e-01	[ 8.3595701e-01]	 2.1055532e-01
      28	 8.1668764e-01	 2.0571813e-01	 8.5722131e-01	 2.0498186e-01	[ 8.7008758e-01]	 1.7631578e-01


.. parsed-literal::

      29	 8.4453135e-01	 2.0722490e-01	 8.8534689e-01	 2.0772540e-01	[ 8.9498797e-01]	 2.0994949e-01


.. parsed-literal::

      30	 8.7666393e-01	 2.0338568e-01	 9.1929909e-01	 2.0497702e-01	[ 9.2130120e-01]	 2.0209980e-01
      31	 8.9731573e-01	 1.9899263e-01	 9.4053632e-01	 1.9750916e-01	[ 9.4481917e-01]	 1.9445300e-01


.. parsed-literal::

      32	 9.1762129e-01	 1.9604998e-01	 9.6199267e-01	 1.9457818e-01	[ 9.5680253e-01]	 2.0503283e-01


.. parsed-literal::

      33	 9.3289970e-01	 1.9417141e-01	 9.7722964e-01	 1.9295208e-01	[ 9.7152914e-01]	 2.1993470e-01


.. parsed-literal::

      34	 9.5613895e-01	 1.9008701e-01	 1.0012564e+00	 1.8857745e-01	[ 9.9400768e-01]	 2.1248507e-01


.. parsed-literal::

      35	 9.9042233e-01	 1.8475050e-01	 1.0369024e+00	 1.8212349e-01	[ 1.0379939e+00]	 2.0958447e-01


.. parsed-literal::

      36	 1.0068437e+00	 1.8203051e-01	 1.0542070e+00	 1.7940311e-01	[ 1.0448863e+00]	 2.1025038e-01


.. parsed-literal::

      37	 1.0185233e+00	 1.8118045e-01	 1.0656152e+00	 1.7854155e-01	[ 1.0627829e+00]	 2.0304203e-01
      38	 1.0325218e+00	 1.7976636e-01	 1.0798078e+00	 1.7694763e-01	[ 1.0814916e+00]	 1.9600272e-01


.. parsed-literal::

      39	 1.0497573e+00	 1.7792914e-01	 1.0968558e+00	 1.7488628e-01	[ 1.1036723e+00]	 1.9785357e-01


.. parsed-literal::

      40	 1.0730353e+00	 1.7092646e-01	 1.1215835e+00	 1.6733973e-01	[ 1.1360619e+00]	 2.1503639e-01


.. parsed-literal::

      41	 1.0911998e+00	 1.6775648e-01	 1.1388480e+00	 1.6342729e-01	[ 1.1571161e+00]	 2.1259046e-01


.. parsed-literal::

      42	 1.1041142e+00	 1.6623846e-01	 1.1515975e+00	 1.6188091e-01	[ 1.1674775e+00]	 2.1745229e-01


.. parsed-literal::

      43	 1.1207913e+00	 1.6239110e-01	 1.1691364e+00	 1.5752341e-01	[ 1.1766436e+00]	 2.0241761e-01
      44	 1.1331630e+00	 1.6009057e-01	 1.1819861e+00	 1.5538581e-01	[ 1.1849726e+00]	 1.8875122e-01


.. parsed-literal::

      45	 1.1446722e+00	 1.5871369e-01	 1.1934504e+00	 1.5365328e-01	[ 1.1965036e+00]	 1.8543744e-01
      46	 1.1563197e+00	 1.5721674e-01	 1.2053358e+00	 1.5201763e-01	[ 1.2042982e+00]	 1.8837643e-01


.. parsed-literal::

      47	 1.1697253e+00	 1.5547268e-01	 1.2190604e+00	 1.5046033e-01	[ 1.2117622e+00]	 1.9050217e-01


.. parsed-literal::

      48	 1.1797716e+00	 1.5298062e-01	 1.2294839e+00	 1.4789345e-01	  1.2105055e+00 	 2.1969843e-01


.. parsed-literal::

      49	 1.1932895e+00	 1.5268077e-01	 1.2429367e+00	 1.4831095e-01	[ 1.2267871e+00]	 2.1004128e-01


.. parsed-literal::

      50	 1.2031220e+00	 1.5094942e-01	 1.2532092e+00	 1.4682646e-01	[ 1.2307284e+00]	 2.0372295e-01
      51	 1.2143298e+00	 1.4959531e-01	 1.2648694e+00	 1.4629546e-01	[ 1.2357276e+00]	 1.8323088e-01


.. parsed-literal::

      52	 1.2256446e+00	 1.4829461e-01	 1.2763242e+00	 1.4658706e-01	[ 1.2357394e+00]	 1.8144894e-01


.. parsed-literal::

      53	 1.2367901e+00	 1.4707345e-01	 1.2875993e+00	 1.4557494e-01	[ 1.2474421e+00]	 2.1114850e-01
      54	 1.2435243e+00	 1.4685944e-01	 1.2942064e+00	 1.4515284e-01	[ 1.2591674e+00]	 2.0281315e-01


.. parsed-literal::

      55	 1.2515145e+00	 1.4600990e-01	 1.3022832e+00	 1.4433233e-01	[ 1.2672755e+00]	 1.7583156e-01


.. parsed-literal::

      56	 1.2639905e+00	 1.4482457e-01	 1.3146960e+00	 1.4337383e-01	[ 1.2834175e+00]	 2.1471667e-01
      57	 1.2725430e+00	 1.4238864e-01	 1.3237714e+00	 1.4146277e-01	[ 1.2842673e+00]	 1.7872429e-01


.. parsed-literal::

      58	 1.2847811e+00	 1.4149760e-01	 1.3358224e+00	 1.4078272e-01	[ 1.2990131e+00]	 1.9747519e-01
      59	 1.2921291e+00	 1.4060535e-01	 1.3433828e+00	 1.4035405e-01	[ 1.3041716e+00]	 1.7352176e-01


.. parsed-literal::

      60	 1.3019008e+00	 1.3931040e-01	 1.3534170e+00	 1.3945519e-01	[ 1.3092295e+00]	 1.9924974e-01
      61	 1.3086057e+00	 1.3851824e-01	 1.3610375e+00	 1.3919197e-01	  1.3031286e+00 	 2.0130062e-01


.. parsed-literal::

      62	 1.3222506e+00	 1.3758116e-01	 1.3743166e+00	 1.3792026e-01	[ 1.3157111e+00]	 1.8531346e-01


.. parsed-literal::

      63	 1.3296135e+00	 1.3691413e-01	 1.3816559e+00	 1.3698766e-01	[ 1.3211482e+00]	 2.0918298e-01
      64	 1.3381295e+00	 1.3626091e-01	 1.3903325e+00	 1.3608504e-01	[ 1.3276935e+00]	 1.9909215e-01


.. parsed-literal::

      65	 1.3415321e+00	 1.3576321e-01	 1.3936639e+00	 1.3557272e-01	[ 1.3298942e+00]	 1.8256497e-01
      66	 1.3510652e+00	 1.3522161e-01	 1.4032257e+00	 1.3525739e-01	[ 1.3372523e+00]	 1.7235065e-01


.. parsed-literal::

      67	 1.3548984e+00	 1.3513653e-01	 1.4071962e+00	 1.3527134e-01	[ 1.3399815e+00]	 2.1576905e-01
      68	 1.3604278e+00	 1.3485878e-01	 1.4130141e+00	 1.3535252e-01	[ 1.3411498e+00]	 1.9804692e-01


.. parsed-literal::

      69	 1.3679454e+00	 1.3460143e-01	 1.4207527e+00	 1.3549297e-01	  1.3405895e+00 	 2.1170568e-01


.. parsed-literal::

      70	 1.3742644e+00	 1.3389133e-01	 1.4275842e+00	 1.3590714e-01	  1.3247124e+00 	 2.1340203e-01


.. parsed-literal::

      71	 1.3818902e+00	 1.3362932e-01	 1.4349471e+00	 1.3555000e-01	  1.3332398e+00 	 2.1227980e-01


.. parsed-literal::

      72	 1.3865409e+00	 1.3318433e-01	 1.4393855e+00	 1.3513782e-01	  1.3368509e+00 	 2.1050143e-01


.. parsed-literal::

      73	 1.3918003e+00	 1.3211503e-01	 1.4448273e+00	 1.3422315e-01	  1.3357077e+00 	 2.0492411e-01


.. parsed-literal::

      74	 1.3961067e+00	 1.3148377e-01	 1.4493691e+00	 1.3419702e-01	  1.3307581e+00 	 2.0942092e-01
      75	 1.4010892e+00	 1.3073586e-01	 1.4542482e+00	 1.3331496e-01	  1.3368865e+00 	 1.9103169e-01


.. parsed-literal::

      76	 1.4045965e+00	 1.3026288e-01	 1.4577947e+00	 1.3302805e-01	  1.3379531e+00 	 2.1367955e-01
      77	 1.4091154e+00	 1.2957263e-01	 1.4624861e+00	 1.3264724e-01	  1.3376825e+00 	 1.7848158e-01


.. parsed-literal::

      78	 1.4134053e+00	 1.2844049e-01	 1.4667450e+00	 1.3205291e-01	  1.3400997e+00 	 2.1045971e-01
      79	 1.4191847e+00	 1.2809743e-01	 1.4725592e+00	 1.3178824e-01	[ 1.3417323e+00]	 1.7984724e-01


.. parsed-literal::

      80	 1.4224598e+00	 1.2789186e-01	 1.4758713e+00	 1.3151218e-01	[ 1.3433594e+00]	 1.8461657e-01
      81	 1.4276762e+00	 1.2760105e-01	 1.4812484e+00	 1.3119027e-01	[ 1.3442199e+00]	 1.8163037e-01


.. parsed-literal::

      82	 1.4290667e+00	 1.2714972e-01	 1.4829429e+00	 1.3051242e-01	  1.3429578e+00 	 2.1441197e-01
      83	 1.4348297e+00	 1.2714347e-01	 1.4884672e+00	 1.3060183e-01	[ 1.3473821e+00]	 1.7613673e-01


.. parsed-literal::

      84	 1.4373199e+00	 1.2695262e-01	 1.4909588e+00	 1.3054523e-01	  1.3470629e+00 	 1.8408298e-01
      85	 1.4413011e+00	 1.2662955e-01	 1.4950131e+00	 1.3030418e-01	  1.3455278e+00 	 1.9781208e-01


.. parsed-literal::

      86	 1.4464692e+00	 1.2629558e-01	 1.5002759e+00	 1.3022850e-01	  1.3430425e+00 	 2.1437454e-01


.. parsed-literal::

      87	 1.4501967e+00	 1.2579707e-01	 1.5042889e+00	 1.2970900e-01	  1.3339474e+00 	 2.0362949e-01


.. parsed-literal::

      88	 1.4556425e+00	 1.2571929e-01	 1.5095370e+00	 1.2976502e-01	  1.3410011e+00 	 2.1306324e-01


.. parsed-literal::

      89	 1.4582129e+00	 1.2556808e-01	 1.5121452e+00	 1.2969803e-01	  1.3414249e+00 	 2.1800995e-01


.. parsed-literal::

      90	 1.4624388e+00	 1.2530591e-01	 1.5164724e+00	 1.2962971e-01	  1.3409305e+00 	 2.1564746e-01


.. parsed-literal::

      91	 1.4662524e+00	 1.2497733e-01	 1.5204564e+00	 1.2938142e-01	  1.3376556e+00 	 2.1688199e-01
      92	 1.4714773e+00	 1.2478830e-01	 1.5256144e+00	 1.2940744e-01	  1.3397731e+00 	 1.9823861e-01


.. parsed-literal::

      93	 1.4742780e+00	 1.2470949e-01	 1.5283965e+00	 1.2930273e-01	  1.3410423e+00 	 2.0683789e-01


.. parsed-literal::

      94	 1.4775904e+00	 1.2453613e-01	 1.5318535e+00	 1.2911861e-01	  1.3379606e+00 	 2.0527434e-01
      95	 1.4792489e+00	 1.2469746e-01	 1.5337700e+00	 1.2922635e-01	  1.3305261e+00 	 1.9910359e-01


.. parsed-literal::

      96	 1.4830309e+00	 1.2451211e-01	 1.5374178e+00	 1.2898498e-01	  1.3348726e+00 	 2.1586084e-01
      97	 1.4849039e+00	 1.2450244e-01	 1.5392716e+00	 1.2902090e-01	  1.3349363e+00 	 1.8550968e-01


.. parsed-literal::

      98	 1.4869957e+00	 1.2453652e-01	 1.5413555e+00	 1.2905453e-01	  1.3352657e+00 	 1.8167019e-01
      99	 1.4908817e+00	 1.2460889e-01	 1.5453130e+00	 1.2927324e-01	  1.3338771e+00 	 1.9664979e-01


.. parsed-literal::

     100	 1.4926812e+00	 1.2473557e-01	 1.5471879e+00	 1.2948815e-01	  1.3347038e+00 	 3.3033180e-01


.. parsed-literal::

     101	 1.4951882e+00	 1.2470584e-01	 1.5497489e+00	 1.2956795e-01	  1.3332778e+00 	 2.2112131e-01


.. parsed-literal::

     102	 1.4972054e+00	 1.2462613e-01	 1.5518041e+00	 1.2960987e-01	  1.3321875e+00 	 2.1971750e-01


.. parsed-literal::

     103	 1.5003111e+00	 1.2444574e-01	 1.5549811e+00	 1.2962873e-01	  1.3286705e+00 	 2.0863461e-01
     104	 1.5019061e+00	 1.2431568e-01	 1.5566751e+00	 1.2999092e-01	  1.3232807e+00 	 1.7903519e-01


.. parsed-literal::

     105	 1.5048997e+00	 1.2411826e-01	 1.5595886e+00	 1.2970492e-01	  1.3244539e+00 	 2.0841479e-01


.. parsed-literal::

     106	 1.5069344e+00	 1.2397185e-01	 1.5615969e+00	 1.2955243e-01	  1.3242687e+00 	 2.0553017e-01


.. parsed-literal::

     107	 1.5089798e+00	 1.2378190e-01	 1.5636509e+00	 1.2944793e-01	  1.3231118e+00 	 2.0670414e-01


.. parsed-literal::

     108	 1.5114914e+00	 1.2368105e-01	 1.5662925e+00	 1.2967820e-01	  1.3152129e+00 	 2.1585417e-01
     109	 1.5141123e+00	 1.2336564e-01	 1.5689033e+00	 1.2954209e-01	  1.3125162e+00 	 2.0188808e-01


.. parsed-literal::

     110	 1.5155780e+00	 1.2335831e-01	 1.5703569e+00	 1.2965976e-01	  1.3117760e+00 	 2.1203780e-01


.. parsed-literal::

     111	 1.5182538e+00	 1.2344033e-01	 1.5730959e+00	 1.3000852e-01	  1.3065037e+00 	 2.1185350e-01


.. parsed-literal::

     112	 1.5197782e+00	 1.2351831e-01	 1.5746727e+00	 1.3037601e-01	  1.2987188e+00 	 2.1362305e-01
     113	 1.5220919e+00	 1.2364535e-01	 1.5769594e+00	 1.3055290e-01	  1.2978824e+00 	 1.9285345e-01


.. parsed-literal::

     114	 1.5237463e+00	 1.2367530e-01	 1.5785922e+00	 1.3063563e-01	  1.2960280e+00 	 1.8246007e-01
     115	 1.5251110e+00	 1.2366102e-01	 1.5799729e+00	 1.3070480e-01	  1.2949208e+00 	 1.8155956e-01


.. parsed-literal::

     116	 1.5270565e+00	 1.2364952e-01	 1.5820336e+00	 1.3094526e-01	  1.2921276e+00 	 2.1144986e-01
     117	 1.5289917e+00	 1.2353445e-01	 1.5840295e+00	 1.3095905e-01	  1.2926233e+00 	 1.8060136e-01


.. parsed-literal::

     118	 1.5303109e+00	 1.2342557e-01	 1.5853714e+00	 1.3092630e-01	  1.2939096e+00 	 2.0343733e-01
     119	 1.5326090e+00	 1.2320869e-01	 1.5877672e+00	 1.3098056e-01	  1.2926378e+00 	 2.0009422e-01


.. parsed-literal::

     120	 1.5343785e+00	 1.2298494e-01	 1.5896508e+00	 1.3101950e-01	  1.2877559e+00 	 2.0687127e-01


.. parsed-literal::

     121	 1.5360810e+00	 1.2291291e-01	 1.5913314e+00	 1.3111900e-01	  1.2879216e+00 	 2.2060680e-01


.. parsed-literal::

     122	 1.5379686e+00	 1.2296464e-01	 1.5932032e+00	 1.3140434e-01	  1.2844229e+00 	 2.1188974e-01
     123	 1.5393406e+00	 1.2304967e-01	 1.5945458e+00	 1.3148481e-01	  1.2870120e+00 	 2.0205331e-01


.. parsed-literal::

     124	 1.5409091e+00	 1.2307395e-01	 1.5960969e+00	 1.3158313e-01	  1.2858109e+00 	 1.9085002e-01


.. parsed-literal::

     125	 1.5426033e+00	 1.2316321e-01	 1.5978306e+00	 1.3169321e-01	  1.2848493e+00 	 2.1711969e-01


.. parsed-literal::

     126	 1.5437070e+00	 1.2312955e-01	 1.5989477e+00	 1.3170900e-01	  1.2841797e+00 	 2.0337009e-01
     127	 1.5453843e+00	 1.2308710e-01	 1.6006674e+00	 1.3171095e-01	  1.2836368e+00 	 1.9334912e-01


.. parsed-literal::

     128	 1.5474574e+00	 1.2290572e-01	 1.6028560e+00	 1.3165621e-01	  1.2799186e+00 	 1.9632578e-01


.. parsed-literal::

     129	 1.5484487e+00	 1.2279475e-01	 1.6039337e+00	 1.3162185e-01	  1.2756744e+00 	 2.1743727e-01


.. parsed-literal::

     130	 1.5497198e+00	 1.2275526e-01	 1.6051196e+00	 1.3159818e-01	  1.2796077e+00 	 2.1196604e-01


.. parsed-literal::

     131	 1.5506915e+00	 1.2275720e-01	 1.6060742e+00	 1.3165444e-01	  1.2797818e+00 	 2.1433878e-01


.. parsed-literal::

     132	 1.5519479e+00	 1.2276224e-01	 1.6073486e+00	 1.3172252e-01	  1.2803124e+00 	 2.0406890e-01


.. parsed-literal::

     133	 1.5530122e+00	 1.2289421e-01	 1.6085407e+00	 1.3181993e-01	  1.2716701e+00 	 2.1065092e-01
     134	 1.5547557e+00	 1.2291632e-01	 1.6102613e+00	 1.3189247e-01	  1.2735663e+00 	 1.9964910e-01


.. parsed-literal::

     135	 1.5562061e+00	 1.2299019e-01	 1.6117447e+00	 1.3198464e-01	  1.2709955e+00 	 2.0109153e-01
     136	 1.5574182e+00	 1.2308147e-01	 1.6130083e+00	 1.3206864e-01	  1.2658220e+00 	 1.8155909e-01


.. parsed-literal::

     137	 1.5590610e+00	 1.2314222e-01	 1.6147967e+00	 1.3226500e-01	  1.2500751e+00 	 2.0824265e-01


.. parsed-literal::

     138	 1.5605419e+00	 1.2325415e-01	 1.6163473e+00	 1.3223858e-01	  1.2430102e+00 	 2.1409106e-01
     139	 1.5614455e+00	 1.2313480e-01	 1.6171993e+00	 1.3216467e-01	  1.2457660e+00 	 1.9406676e-01


.. parsed-literal::

     140	 1.5624183e+00	 1.2305454e-01	 1.6181687e+00	 1.3211485e-01	  1.2441547e+00 	 2.0156240e-01
     141	 1.5636080e+00	 1.2298934e-01	 1.6193749e+00	 1.3208588e-01	  1.2416286e+00 	 1.8474507e-01


.. parsed-literal::

     142	 1.5648008e+00	 1.2301134e-01	 1.6206881e+00	 1.3204446e-01	  1.2276281e+00 	 2.1531701e-01
     143	 1.5661602e+00	 1.2303895e-01	 1.6220458e+00	 1.3209174e-01	  1.2269028e+00 	 1.8222117e-01


.. parsed-literal::

     144	 1.5670734e+00	 1.2309568e-01	 1.6229989e+00	 1.3213344e-01	  1.2228120e+00 	 1.8718481e-01


.. parsed-literal::

     145	 1.5680003e+00	 1.2307744e-01	 1.6239942e+00	 1.3213772e-01	  1.2169871e+00 	 2.0243120e-01


.. parsed-literal::

     146	 1.5690028e+00	 1.2303383e-01	 1.6252070e+00	 1.3214319e-01	  1.1959884e+00 	 2.1092463e-01
     147	 1.5700667e+00	 1.2291815e-01	 1.6262420e+00	 1.3209132e-01	  1.1983204e+00 	 1.9909883e-01


.. parsed-literal::

     148	 1.5708312e+00	 1.2284454e-01	 1.6270161e+00	 1.3208129e-01	  1.1941522e+00 	 1.9716454e-01


.. parsed-literal::

     149	 1.5719183e+00	 1.2279808e-01	 1.6281042e+00	 1.3210575e-01	  1.1870761e+00 	 2.1449518e-01


.. parsed-literal::

     150	 1.5723710e+00	 1.2280206e-01	 1.6286083e+00	 1.3228341e-01	  1.1619885e+00 	 2.0693111e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.15 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f694af9e110>



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
    CPU times: user 1.73 s, sys: 33 ms, total: 1.76 s
    Wall time: 541 ms


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

