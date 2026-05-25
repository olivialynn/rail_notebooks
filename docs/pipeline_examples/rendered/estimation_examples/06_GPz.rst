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
       1	-3.4291433e-01	 3.1955039e-01	-3.3270263e-01	 3.2400276e-01	[-3.4160027e-01]	 4.4870710e-01


.. parsed-literal::

       2	-2.6911105e-01	 3.0857739e-01	-2.4432408e-01	 3.1357366e-01	[-2.6028429e-01]	 2.2263813e-01


.. parsed-literal::

       3	-2.2116387e-01	 2.8602917e-01	-1.7631045e-01	 2.9297376e-01	[-2.0807350e-01]	 2.6830053e-01


.. parsed-literal::

       4	-1.7882147e-01	 2.6757966e-01	-1.2828813e-01	 2.7759531e-01	[-1.7734797e-01]	 2.8584743e-01
       5	-1.1897938e-01	 2.5555620e-01	-8.9593327e-02	 2.6298172e-01	[-1.2750080e-01]	 2.0214677e-01


.. parsed-literal::

       6	-6.7137828e-02	 2.5158256e-01	-4.0098272e-02	 2.5812244e-01	[-6.3174247e-02]	 1.7556620e-01
       7	-4.6638435e-02	 2.4748438e-01	-2.3516849e-02	 2.5420033e-01	[-4.8485931e-02]	 1.9440365e-01


.. parsed-literal::

       8	-3.6080801e-02	 2.4574731e-01	-1.5789843e-02	 2.5280430e-01	[-4.4827182e-02]	 2.0129108e-01
       9	-2.3480926e-02	 2.4334337e-01	-5.1359830e-03	 2.5007450e-01	[-3.1455490e-02]	 1.7872834e-01


.. parsed-literal::

      10	-1.3716467e-02	 2.4140150e-01	 2.7830492e-03	 2.4713238e-01	[-1.9840224e-02]	 2.0232105e-01


.. parsed-literal::

      11	-8.0793398e-03	 2.4033730e-01	 7.6304642e-03	 2.4575373e-01	[-1.3243676e-02]	 2.0151353e-01
      12	-2.7389750e-03	 2.3945019e-01	 1.2044410e-02	 2.4476512e-01	[-7.6867498e-03]	 1.9929385e-01


.. parsed-literal::

      13	 2.2089468e-03	 2.3847897e-01	 1.6982462e-02	 2.4374656e-01	[-2.7164865e-03]	 1.9869375e-01


.. parsed-literal::

      14	 1.0191941e-01	 2.2441567e-01	 1.2504674e-01	 2.4003303e-01	[ 9.2944397e-02]	 4.2422390e-01
      15	 1.4243890e-01	 2.2339405e-01	 1.6536705e-01	 2.3098819e-01	[ 1.4079991e-01]	 1.9993424e-01


.. parsed-literal::

      16	 2.1942478e-01	 2.1958073e-01	 2.4544861e-01	 2.2648558e-01	[ 2.1953855e-01]	 2.0015931e-01


.. parsed-literal::

      17	 2.9261269e-01	 2.1697979e-01	 3.2441632e-01	 2.1876698e-01	[ 3.0825705e-01]	 2.0416021e-01
      18	 3.5199100e-01	 2.1229505e-01	 3.8581022e-01	 2.1565416e-01	[ 3.6287262e-01]	 1.9739151e-01


.. parsed-literal::

      19	 4.2969521e-01	 2.0898686e-01	 4.6613206e-01	 2.1348663e-01	[ 4.4017419e-01]	 1.9391489e-01
      20	 4.9331744e-01	 2.0293610e-01	 5.3171050e-01	 2.0449593e-01	[ 5.1025177e-01]	 1.9366932e-01


.. parsed-literal::

      21	 5.6841116e-01	 1.9580487e-01	 6.0996573e-01	 1.9227943e-01	[ 5.9426986e-01]	 1.9689512e-01
      22	 6.2540800e-01	 1.8843529e-01	 6.6882507e-01	 1.8468565e-01	[ 6.4226878e-01]	 1.9900393e-01


.. parsed-literal::

      23	 6.6757354e-01	 1.8423598e-01	 7.1076250e-01	 1.8031365e-01	[ 6.8020286e-01]	 1.9813108e-01
      24	 7.2635979e-01	 1.8265386e-01	 7.6815233e-01	 1.7840645e-01	[ 7.3570018e-01]	 2.0264030e-01


.. parsed-literal::

      25	 7.6018566e-01	 1.8791969e-01	 8.0172493e-01	 1.8367557e-01	[ 7.5961621e-01]	 2.0160723e-01
      26	 7.9018285e-01	 1.8315338e-01	 8.3137491e-01	 1.7951895e-01	[ 7.8633905e-01]	 1.9915199e-01


.. parsed-literal::

      27	 8.1440644e-01	 1.8135364e-01	 8.5631495e-01	 1.7771030e-01	[ 8.1049930e-01]	 1.8044543e-01


.. parsed-literal::

      28	 8.6271435e-01	 1.7683249e-01	 9.0497072e-01	 1.7184485e-01	[ 8.6872371e-01]	 2.1230483e-01


.. parsed-literal::

      29	 8.8807426e-01	 1.7589775e-01	 9.3133845e-01	 1.6924500e-01	[ 8.9652648e-01]	 2.0619678e-01


.. parsed-literal::

      30	 9.0859448e-01	 1.7224210e-01	 9.5239853e-01	 1.6493552e-01	[ 9.1972664e-01]	 2.0595026e-01


.. parsed-literal::

      31	 9.2812403e-01	 1.7009895e-01	 9.7207107e-01	 1.6325330e-01	[ 9.3561103e-01]	 2.1068692e-01


.. parsed-literal::

      32	 9.5127306e-01	 1.6780033e-01	 9.9620788e-01	 1.6180630e-01	[ 9.6164018e-01]	 2.0308685e-01
      33	 9.7568276e-01	 1.6468920e-01	 1.0214960e+00	 1.5864640e-01	[ 9.7857138e-01]	 2.0274043e-01


.. parsed-literal::

      34	 9.9342996e-01	 1.6177772e-01	 1.0399450e+00	 1.5586105e-01	[ 9.9674811e-01]	 1.7807150e-01


.. parsed-literal::

      35	 1.0089395e+00	 1.6094821e-01	 1.0554697e+00	 1.5442103e-01	[ 1.0086038e+00]	 2.1260047e-01
      36	 1.0260057e+00	 1.5978488e-01	 1.0734100e+00	 1.5262124e-01	[ 1.0186640e+00]	 1.9352198e-01


.. parsed-literal::

      37	 1.0384819e+00	 1.5867685e-01	 1.0862988e+00	 1.5142308e-01	[ 1.0300193e+00]	 1.9763517e-01


.. parsed-literal::

      38	 1.0556252e+00	 1.5640611e-01	 1.1050739e+00	 1.5064504e-01	[ 1.0440169e+00]	 2.0799661e-01


.. parsed-literal::

      39	 1.0740121e+00	 1.5645391e-01	 1.1230390e+00	 1.5051044e-01	[ 1.0623165e+00]	 2.1820807e-01
      40	 1.0797279e+00	 1.5587558e-01	 1.1286073e+00	 1.5016564e-01	[ 1.0671763e+00]	 1.8652940e-01


.. parsed-literal::

      41	 1.0994311e+00	 1.5425528e-01	 1.1486145e+00	 1.5056376e-01	[ 1.0824783e+00]	 2.1481133e-01


.. parsed-literal::

      42	 1.1114479e+00	 1.5296889e-01	 1.1605748e+00	 1.5078670e-01	[ 1.0856422e+00]	 2.0320368e-01
      43	 1.1253560e+00	 1.5114857e-01	 1.1746292e+00	 1.4884261e-01	[ 1.1018877e+00]	 1.9403386e-01


.. parsed-literal::

      44	 1.1385113e+00	 1.4980429e-01	 1.1884837e+00	 1.4689893e-01	[ 1.1082018e+00]	 2.0492721e-01


.. parsed-literal::

      45	 1.1483827e+00	 1.4907350e-01	 1.1988620e+00	 1.4533717e-01	[ 1.1162982e+00]	 2.0105076e-01


.. parsed-literal::

      46	 1.1630497e+00	 1.4769088e-01	 1.2143381e+00	 1.4394897e-01	[ 1.1194103e+00]	 2.1511102e-01
      47	 1.1754231e+00	 1.4655272e-01	 1.2266135e+00	 1.4229750e-01	[ 1.1294412e+00]	 1.7221856e-01


.. parsed-literal::

      48	 1.1839346e+00	 1.4610331e-01	 1.2347137e+00	 1.4140591e-01	[ 1.1370335e+00]	 2.0073962e-01


.. parsed-literal::

      49	 1.1929280e+00	 1.4539025e-01	 1.2433813e+00	 1.4061037e-01	[ 1.1436434e+00]	 2.0957994e-01
      50	 1.2040365e+00	 1.4476009e-01	 1.2548013e+00	 1.3905058e-01	[ 1.1491870e+00]	 1.9584060e-01


.. parsed-literal::

      51	 1.2145994e+00	 1.4401404e-01	 1.2655155e+00	 1.3789168e-01	[ 1.1548490e+00]	 2.1222448e-01


.. parsed-literal::

      52	 1.2248668e+00	 1.4285372e-01	 1.2761952e+00	 1.3656702e-01	[ 1.1587521e+00]	 2.0493078e-01
      53	 1.2346064e+00	 1.4232095e-01	 1.2864210e+00	 1.3616340e-01	  1.1549200e+00 	 1.9648123e-01


.. parsed-literal::

      54	 1.2435006e+00	 1.4113973e-01	 1.2955629e+00	 1.3423673e-01	  1.1582687e+00 	 1.9545817e-01


.. parsed-literal::

      55	 1.2511467e+00	 1.4069763e-01	 1.3029315e+00	 1.3353793e-01	[ 1.1681743e+00]	 2.0812535e-01


.. parsed-literal::

      56	 1.2606956e+00	 1.4044825e-01	 1.3125415e+00	 1.3296136e-01	[ 1.1734575e+00]	 2.0433593e-01


.. parsed-literal::

      57	 1.2705410e+00	 1.4098461e-01	 1.3226643e+00	 1.3267137e-01	[ 1.1804307e+00]	 2.0669484e-01
      58	 1.2795762e+00	 1.4048294e-01	 1.3319015e+00	 1.3207345e-01	[ 1.1875941e+00]	 1.9840169e-01


.. parsed-literal::

      59	 1.2881638e+00	 1.4002885e-01	 1.3408239e+00	 1.3141185e-01	[ 1.1920172e+00]	 2.0268035e-01


.. parsed-literal::

      60	 1.2992546e+00	 1.3933280e-01	 1.3524114e+00	 1.3013093e-01	[ 1.2039550e+00]	 2.1684074e-01


.. parsed-literal::

      61	 1.3077801e+00	 1.3836882e-01	 1.3612346e+00	 1.2843516e-01	  1.2018775e+00 	 2.1861959e-01


.. parsed-literal::

      62	 1.3164898e+00	 1.3812200e-01	 1.3697066e+00	 1.2803450e-01	[ 1.2149015e+00]	 2.1497893e-01
      63	 1.3245175e+00	 1.3783857e-01	 1.3776344e+00	 1.2754011e-01	[ 1.2230130e+00]	 1.8621039e-01


.. parsed-literal::

      64	 1.3324302e+00	 1.3792577e-01	 1.3857344e+00	 1.2721068e-01	[ 1.2252450e+00]	 1.9449520e-01
      65	 1.3368694e+00	 1.3777680e-01	 1.3905735e+00	 1.2826510e-01	  1.2147740e+00 	 1.9173145e-01


.. parsed-literal::

      66	 1.3453794e+00	 1.3767294e-01	 1.3989277e+00	 1.2764399e-01	[ 1.2272636e+00]	 2.0386696e-01


.. parsed-literal::

      67	 1.3490044e+00	 1.3769510e-01	 1.4025891e+00	 1.2775431e-01	[ 1.2294110e+00]	 2.1420074e-01


.. parsed-literal::

      68	 1.3557477e+00	 1.3786130e-01	 1.4095814e+00	 1.2834491e-01	[ 1.2308057e+00]	 2.0945573e-01


.. parsed-literal::

      69	 1.3631953e+00	 1.3796066e-01	 1.4172410e+00	 1.2927858e-01	  1.2195623e+00 	 2.0720553e-01
      70	 1.3699592e+00	 1.3804557e-01	 1.4240509e+00	 1.2957323e-01	  1.2289922e+00 	 1.9528937e-01


.. parsed-literal::

      71	 1.3751982e+00	 1.3800275e-01	 1.4293489e+00	 1.2986963e-01	[ 1.2364858e+00]	 2.1186352e-01
      72	 1.3803325e+00	 1.3786331e-01	 1.4345438e+00	 1.2994417e-01	[ 1.2414142e+00]	 1.8016768e-01


.. parsed-literal::

      73	 1.3865476e+00	 1.3759821e-01	 1.4409313e+00	 1.3037168e-01	[ 1.2438268e+00]	 2.1175241e-01


.. parsed-literal::

      74	 1.3924604e+00	 1.3734580e-01	 1.4469102e+00	 1.3042563e-01	[ 1.2445465e+00]	 2.1545959e-01
      75	 1.3990492e+00	 1.3716729e-01	 1.4536372e+00	 1.3045548e-01	  1.2419295e+00 	 1.9814754e-01


.. parsed-literal::

      76	 1.4028883e+00	 1.3725414e-01	 1.4575874e+00	 1.3094346e-01	  1.2440218e+00 	 2.0200706e-01


.. parsed-literal::

      77	 1.4071449e+00	 1.3723198e-01	 1.4617964e+00	 1.3085800e-01	[ 1.2453682e+00]	 2.1071124e-01
      78	 1.4130726e+00	 1.3721103e-01	 1.4678928e+00	 1.3087176e-01	  1.2435124e+00 	 1.9883847e-01


.. parsed-literal::

      79	 1.4179911e+00	 1.3712387e-01	 1.4729706e+00	 1.3087528e-01	  1.2393050e+00 	 1.6974592e-01
      80	 1.4234681e+00	 1.3659380e-01	 1.4790091e+00	 1.3107242e-01	  1.2133415e+00 	 1.9702721e-01


.. parsed-literal::

      81	 1.4296052e+00	 1.3663785e-01	 1.4849955e+00	 1.3109714e-01	  1.2185470e+00 	 2.0179319e-01


.. parsed-literal::

      82	 1.4325329e+00	 1.3634836e-01	 1.4877517e+00	 1.3087008e-01	  1.2244793e+00 	 2.0417094e-01


.. parsed-literal::

      83	 1.4368574e+00	 1.3612944e-01	 1.4920669e+00	 1.3106376e-01	  1.2164075e+00 	 2.0229936e-01


.. parsed-literal::

      84	 1.4408547e+00	 1.3585591e-01	 1.4960764e+00	 1.3100497e-01	  1.2187807e+00 	 2.0653844e-01
      85	 1.4438906e+00	 1.3577769e-01	 1.4991241e+00	 1.3093824e-01	  1.2218691e+00 	 1.9814777e-01


.. parsed-literal::

      86	 1.4495474e+00	 1.3552449e-01	 1.5049732e+00	 1.3090512e-01	  1.2182059e+00 	 2.0025635e-01


.. parsed-literal::

      87	 1.4521792e+00	 1.3538962e-01	 1.5078710e+00	 1.3081500e-01	  1.2414271e+00 	 2.0729756e-01


.. parsed-literal::

      88	 1.4563622e+00	 1.3534448e-01	 1.5118168e+00	 1.3065024e-01	  1.2421506e+00 	 2.1257305e-01


.. parsed-literal::

      89	 1.4599697e+00	 1.3533335e-01	 1.5154071e+00	 1.3064729e-01	  1.2403817e+00 	 2.0705128e-01
      90	 1.4629641e+00	 1.3535051e-01	 1.5184402e+00	 1.3055691e-01	  1.2435669e+00 	 1.9588017e-01


.. parsed-literal::

      91	 1.4676465e+00	 1.3526096e-01	 1.5233199e+00	 1.3005814e-01	[ 1.2464533e+00]	 2.0106697e-01


.. parsed-literal::

      92	 1.4717289e+00	 1.3552072e-01	 1.5274699e+00	 1.3010779e-01	  1.2397667e+00 	 2.1875143e-01


.. parsed-literal::

      93	 1.4740120e+00	 1.3536992e-01	 1.5297789e+00	 1.2993221e-01	  1.2402767e+00 	 2.1533442e-01
      94	 1.4778997e+00	 1.3506957e-01	 1.5338601e+00	 1.2970934e-01	  1.2318062e+00 	 1.8861508e-01


.. parsed-literal::

      95	 1.4814088e+00	 1.3488098e-01	 1.5375225e+00	 1.2936197e-01	  1.2246437e+00 	 1.9910502e-01


.. parsed-literal::

      96	 1.4854909e+00	 1.3464219e-01	 1.5416560e+00	 1.2922860e-01	  1.2171733e+00 	 2.0975566e-01


.. parsed-literal::

      97	 1.4882986e+00	 1.3465738e-01	 1.5444283e+00	 1.2902941e-01	  1.2167512e+00 	 2.0520639e-01


.. parsed-literal::

      98	 1.4908400e+00	 1.3449389e-01	 1.5469360e+00	 1.2878477e-01	  1.2202126e+00 	 2.0087719e-01
      99	 1.4935743e+00	 1.3430244e-01	 1.5496497e+00	 1.2849231e-01	  1.2193740e+00 	 2.0147204e-01


.. parsed-literal::

     100	 1.4964611e+00	 1.3382981e-01	 1.5527102e+00	 1.2779749e-01	  1.2243678e+00 	 1.9546199e-01
     101	 1.4993830e+00	 1.3371342e-01	 1.5556890e+00	 1.2761943e-01	  1.2102778e+00 	 1.9641352e-01


.. parsed-literal::

     102	 1.5009317e+00	 1.3367574e-01	 1.5572284e+00	 1.2769817e-01	  1.2117771e+00 	 1.9328952e-01
     103	 1.5045515e+00	 1.3364799e-01	 1.5609365e+00	 1.2777914e-01	  1.2100982e+00 	 1.9240522e-01


.. parsed-literal::

     104	 1.5058319e+00	 1.3350156e-01	 1.5623023e+00	 1.2765240e-01	  1.2115243e+00 	 1.9967127e-01
     105	 1.5089378e+00	 1.3350826e-01	 1.5653215e+00	 1.2763532e-01	  1.2135707e+00 	 1.9356894e-01


.. parsed-literal::

     106	 1.5103756e+00	 1.3344211e-01	 1.5667447e+00	 1.2746498e-01	  1.2113354e+00 	 1.9716620e-01
     107	 1.5130334e+00	 1.3328273e-01	 1.5694166e+00	 1.2721353e-01	  1.2051498e+00 	 1.9625020e-01


.. parsed-literal::

     108	 1.5164541e+00	 1.3316089e-01	 1.5729395e+00	 1.2700651e-01	  1.1834299e+00 	 2.0062160e-01
     109	 1.5190182e+00	 1.3297483e-01	 1.5756297e+00	 1.2707531e-01	  1.1744527e+00 	 1.7277598e-01


.. parsed-literal::

     110	 1.5210814e+00	 1.3299339e-01	 1.5776537e+00	 1.2714767e-01	  1.1754724e+00 	 2.0457840e-01
     111	 1.5235225e+00	 1.3298433e-01	 1.5801680e+00	 1.2724895e-01	  1.1737164e+00 	 1.7676663e-01


.. parsed-literal::

     112	 1.5250758e+00	 1.3308879e-01	 1.5818255e+00	 1.2739807e-01	  1.1634099e+00 	 2.1412563e-01


.. parsed-literal::

     113	 1.5270855e+00	 1.3295599e-01	 1.5838156e+00	 1.2726159e-01	  1.1679217e+00 	 2.1202683e-01


.. parsed-literal::

     114	 1.5292231e+00	 1.3273404e-01	 1.5859694e+00	 1.2698923e-01	  1.1638844e+00 	 2.0740080e-01


.. parsed-literal::

     115	 1.5305158e+00	 1.3257571e-01	 1.5872696e+00	 1.2682714e-01	  1.1650070e+00 	 2.1643758e-01


.. parsed-literal::

     116	 1.5325323e+00	 1.3236264e-01	 1.5892917e+00	 1.2661450e-01	  1.1613134e+00 	 2.0473647e-01
     117	 1.5351081e+00	 1.3204661e-01	 1.5918995e+00	 1.2634214e-01	  1.1609423e+00 	 1.8162560e-01


.. parsed-literal::

     118	 1.5375547e+00	 1.3178492e-01	 1.5944291e+00	 1.2627762e-01	  1.1512533e+00 	 2.1074963e-01


.. parsed-literal::

     119	 1.5396920e+00	 1.3161648e-01	 1.5965564e+00	 1.2615834e-01	  1.1526754e+00 	 3.9129615e-01


.. parsed-literal::

     120	 1.5413565e+00	 1.3149117e-01	 1.5982530e+00	 1.2608882e-01	  1.1466138e+00 	 2.1474552e-01


.. parsed-literal::

     121	 1.5429704e+00	 1.3128940e-01	 1.5999651e+00	 1.2603223e-01	  1.1383375e+00 	 2.0220566e-01
     122	 1.5449281e+00	 1.3097322e-01	 1.6020008e+00	 1.2580768e-01	  1.1165513e+00 	 1.9623065e-01


.. parsed-literal::

     123	 1.5467388e+00	 1.3074089e-01	 1.6038355e+00	 1.2566224e-01	  1.1052782e+00 	 1.8613315e-01


.. parsed-literal::

     124	 1.5485693e+00	 1.3051909e-01	 1.6056851e+00	 1.2550066e-01	  1.0990573e+00 	 2.1483159e-01
     125	 1.5503503e+00	 1.3042836e-01	 1.6075011e+00	 1.2549786e-01	  1.0875744e+00 	 2.0308328e-01


.. parsed-literal::

     126	 1.5521772e+00	 1.3039983e-01	 1.6092969e+00	 1.2555134e-01	  1.0932308e+00 	 2.0229506e-01
     127	 1.5541107e+00	 1.3041218e-01	 1.6112460e+00	 1.2563831e-01	  1.0875707e+00 	 1.9364858e-01


.. parsed-literal::

     128	 1.5555031e+00	 1.3025903e-01	 1.6126976e+00	 1.2570228e-01	  1.0854208e+00 	 1.9413018e-01
     129	 1.5568939e+00	 1.3020160e-01	 1.6141203e+00	 1.2575860e-01	  1.0792674e+00 	 1.9978809e-01


.. parsed-literal::

     130	 1.5584072e+00	 1.3014204e-01	 1.6157101e+00	 1.2575267e-01	  1.0741930e+00 	 2.0924544e-01


.. parsed-literal::

     131	 1.5596406e+00	 1.3011248e-01	 1.6169970e+00	 1.2583894e-01	  1.0708437e+00 	 2.0591140e-01
     132	 1.5608704e+00	 1.3015713e-01	 1.6182374e+00	 1.2591843e-01	  1.0697413e+00 	 1.8007469e-01


.. parsed-literal::

     133	 1.5630874e+00	 1.3006023e-01	 1.6204777e+00	 1.2612380e-01	  1.0603982e+00 	 2.0502448e-01
     134	 1.5644399e+00	 1.3013712e-01	 1.6218643e+00	 1.2640340e-01	  1.0533828e+00 	 2.0002985e-01


.. parsed-literal::

     135	 1.5658730e+00	 1.3002351e-01	 1.6232130e+00	 1.2633194e-01	  1.0533164e+00 	 1.9769239e-01


.. parsed-literal::

     136	 1.5668810e+00	 1.2986580e-01	 1.6242028e+00	 1.2628364e-01	  1.0505599e+00 	 2.0417953e-01


.. parsed-literal::

     137	 1.5679322e+00	 1.2968087e-01	 1.6252564e+00	 1.2621358e-01	  1.0527557e+00 	 2.0367503e-01
     138	 1.5694736e+00	 1.2948682e-01	 1.6268111e+00	 1.2613940e-01	  1.0458400e+00 	 1.8065500e-01


.. parsed-literal::

     139	 1.5708209e+00	 1.2929413e-01	 1.6282046e+00	 1.2600908e-01	  1.0476473e+00 	 2.0184803e-01


.. parsed-literal::

     140	 1.5717838e+00	 1.2931265e-01	 1.6291458e+00	 1.2598599e-01	  1.0464949e+00 	 2.1460700e-01


.. parsed-literal::

     141	 1.5726238e+00	 1.2935406e-01	 1.6299834e+00	 1.2604197e-01	  1.0408765e+00 	 2.1194625e-01


.. parsed-literal::

     142	 1.5735179e+00	 1.2926637e-01	 1.6309129e+00	 1.2595052e-01	  1.0358827e+00 	 2.1572709e-01
     143	 1.5745243e+00	 1.2920150e-01	 1.6319372e+00	 1.2595828e-01	  1.0286257e+00 	 2.0447373e-01


.. parsed-literal::

     144	 1.5758832e+00	 1.2898100e-01	 1.6333573e+00	 1.2586988e-01	  1.0175841e+00 	 1.9977379e-01
     145	 1.5769770e+00	 1.2881735e-01	 1.6345192e+00	 1.2574807e-01	  1.0053575e+00 	 1.9871378e-01


.. parsed-literal::

     146	 1.5782804e+00	 1.2862827e-01	 1.6358512e+00	 1.2556475e-01	  9.9745474e-01 	 1.9933105e-01
     147	 1.5794091e+00	 1.2850793e-01	 1.6370047e+00	 1.2546819e-01	  9.9310459e-01 	 1.9429946e-01


.. parsed-literal::

     148	 1.5806273e+00	 1.2836942e-01	 1.6382553e+00	 1.2530254e-01	  9.8479111e-01 	 1.8620253e-01
     149	 1.5819064e+00	 1.2822633e-01	 1.6395475e+00	 1.2528420e-01	  9.7838227e-01 	 1.9787383e-01


.. parsed-literal::

     150	 1.5830210e+00	 1.2809097e-01	 1.6406808e+00	 1.2524169e-01	  9.5774143e-01 	 1.9876027e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 857 ms, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff5944d3730>



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
    CPU times: user 955 ms, sys: 48.9 ms, total: 1 s
    Wall time: 363 ms


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

