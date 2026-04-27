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
       1	-3.2601958e-01	 3.1508097e-01	-3.1569669e-01	 3.3682948e-01	[-3.5889741e-01]	 4.5840597e-01


.. parsed-literal::

       2	-2.5380614e-01	 3.0358186e-01	-2.2858874e-01	 3.2772813e-01	[-3.0292385e-01]	 2.2816420e-01


.. parsed-literal::

       3	-2.1239632e-01	 2.8529820e-01	-1.7151708e-01	 3.0925487e-01	[-2.7449597e-01]	 2.7575421e-01
       4	-1.7048852e-01	 2.6039696e-01	-1.2744181e-01	 2.8198496e-01	[-2.6205702e-01]	 1.7941308e-01


.. parsed-literal::

       5	-8.7590094e-02	 2.5104245e-01	-5.2952360e-02	 2.7607479e-01	[-1.7987265e-01]	 2.1098852e-01


.. parsed-literal::

       6	-4.6679988e-02	 2.4483653e-01	-1.3542696e-02	 2.6965899e-01	[-1.1101156e-01]	 2.0288873e-01


.. parsed-literal::

       7	-2.7519469e-02	 2.4218364e-01	-1.5093540e-03	 2.6760552e-01	[-1.0888020e-01]	 2.2106242e-01


.. parsed-literal::

       8	-9.5395972e-03	 2.3914588e-01	 1.1564120e-02	 2.6598096e-01	[-1.0284670e-01]	 2.0696783e-01
       9	 1.3721000e-03	 2.3718916e-01	 1.9408514e-02	 2.6401074e-01	 -1.0593936e-01 	 2.0195341e-01


.. parsed-literal::

      10	 9.6393519e-03	 2.3595198e-01	 2.6005114e-02	 2.6096989e-01	[-8.3824458e-02]	 2.1223807e-01


.. parsed-literal::

      11	 1.4030061e-02	 2.3514416e-01	 2.9726208e-02	 2.6011246e-01	[-8.0450335e-02]	 2.0816064e-01


.. parsed-literal::

      12	 1.8040217e-02	 2.3444326e-01	 3.3173114e-02	 2.5933893e-01	[-7.6592302e-02]	 2.0693564e-01
      13	 2.3284046e-02	 2.3343952e-01	 3.8532240e-02	 2.5864646e-01	[-7.5427391e-02]	 1.9772863e-01


.. parsed-literal::

      14	 9.8985411e-02	 2.2177946e-01	 1.1760532e-01	 2.5023409e-01	[ 2.0843915e-02]	 3.2328153e-01


.. parsed-literal::

      15	 1.3180162e-01	 2.1805040e-01	 1.5314688e-01	 2.4546211e-01	[ 7.8997133e-02]	 3.0811334e-01


.. parsed-literal::

      16	 1.9606652e-01	 2.1450995e-01	 2.2016902e-01	 2.4008883e-01	[ 1.4138111e-01]	 2.1831107e-01


.. parsed-literal::

      17	 2.6762292e-01	 2.1032011e-01	 2.9877171e-01	 2.3315721e-01	[ 2.0786056e-01]	 2.1371675e-01


.. parsed-literal::

      18	 2.9363270e-01	 2.1004013e-01	 3.2628417e-01	 2.3128358e-01	[ 2.3935022e-01]	 2.1265006e-01


.. parsed-literal::

      19	 3.3220253e-01	 2.0545527e-01	 3.6620531e-01	 2.2877614e-01	[ 2.8133559e-01]	 2.1054006e-01


.. parsed-literal::

      20	 3.6896131e-01	 2.0251844e-01	 4.0334442e-01	 2.2486555e-01	[ 3.2579590e-01]	 2.1412325e-01


.. parsed-literal::

      21	 4.7679465e-01	 1.9852093e-01	 5.1197760e-01	 2.1973493e-01	[ 4.4810501e-01]	 2.0912242e-01
      22	 5.5978172e-01	 2.0597901e-01	 5.9903119e-01	 2.3887631e-01	[ 5.3264070e-01]	 1.8471837e-01


.. parsed-literal::

      23	 6.1067264e-01	 2.0192578e-01	 6.5046338e-01	 2.2715822e-01	[ 5.9555848e-01]	 1.8935895e-01


.. parsed-literal::

      24	 6.4283825e-01	 1.9801757e-01	 6.8191187e-01	 2.2187043e-01	[ 6.2787936e-01]	 2.0786095e-01


.. parsed-literal::

      25	 6.7658326e-01	 1.9687689e-01	 7.1535212e-01	 2.1926746e-01	[ 6.6097974e-01]	 2.1552515e-01


.. parsed-literal::

      26	 7.0649914e-01	 1.9552960e-01	 7.4448408e-01	 2.1764925e-01	[ 6.8511412e-01]	 3.2317996e-01


.. parsed-literal::

      27	 7.3585684e-01	 1.9311298e-01	 7.7414840e-01	 2.1727532e-01	[ 7.1640056e-01]	 2.1258688e-01


.. parsed-literal::

      28	 7.5671462e-01	 1.9271449e-01	 7.9636017e-01	 2.1666937e-01	[ 7.3922702e-01]	 2.1226716e-01


.. parsed-literal::

      29	 7.7974962e-01	 1.9311004e-01	 8.2038503e-01	 2.1908122e-01	[ 7.6518900e-01]	 2.1254945e-01


.. parsed-literal::

      30	 8.0844354e-01	 1.9295652e-01	 8.5107491e-01	 2.2023973e-01	[ 7.9520461e-01]	 2.1789384e-01
      31	 8.3796268e-01	 1.9239257e-01	 8.8323180e-01	 2.2122135e-01	[ 8.2108044e-01]	 1.9677615e-01


.. parsed-literal::

      32	 8.6545318e-01	 1.9258026e-01	 9.1048667e-01	 2.2024555e-01	[ 8.5095174e-01]	 1.7628431e-01
      33	 8.8369604e-01	 1.8907101e-01	 9.2952444e-01	 2.1674427e-01	[ 8.7016979e-01]	 1.9734573e-01


.. parsed-literal::

      34	 9.1330413e-01	 1.8569477e-01	 9.5793338e-01	 2.1386954e-01	[ 8.9324031e-01]	 2.0823121e-01


.. parsed-literal::

      35	 9.3198979e-01	 1.8419353e-01	 9.7704745e-01	 2.1181225e-01	[ 9.1349232e-01]	 2.0602441e-01
      36	 9.4893997e-01	 1.8184425e-01	 9.9447855e-01	 2.0933736e-01	[ 9.3316792e-01]	 1.7483616e-01


.. parsed-literal::

      37	 9.6967544e-01	 1.7696143e-01	 1.0170672e+00	 2.0468345e-01	[ 9.5376745e-01]	 2.1502280e-01
      38	 9.8604076e-01	 1.7573818e-01	 1.0340018e+00	 2.0718370e-01	[ 9.6740691e-01]	 1.9976544e-01


.. parsed-literal::

      39	 9.9850742e-01	 1.7475751e-01	 1.0465955e+00	 2.0636703e-01	[ 9.7866548e-01]	 2.1816850e-01
      40	 1.0103668e+00	 1.7338507e-01	 1.0591123e+00	 2.0511645e-01	[ 9.9093928e-01]	 1.9984007e-01


.. parsed-literal::

      41	 1.0234373e+00	 1.7245185e-01	 1.0728161e+00	 2.0463470e-01	[ 1.0040778e+00]	 2.0057130e-01


.. parsed-literal::

      42	 1.0371991e+00	 1.7084267e-01	 1.0869829e+00	 2.0221846e-01	[ 1.0181119e+00]	 2.0573401e-01


.. parsed-literal::

      43	 1.0477831e+00	 1.6842011e-01	 1.0977704e+00	 1.9905290e-01	[ 1.0305019e+00]	 2.0714355e-01


.. parsed-literal::

      44	 1.0576395e+00	 1.6676315e-01	 1.1072247e+00	 1.9604154e-01	[ 1.0371768e+00]	 2.0825815e-01


.. parsed-literal::

      45	 1.0672447e+00	 1.6528885e-01	 1.1167832e+00	 1.9430900e-01	[ 1.0422332e+00]	 2.0877743e-01


.. parsed-literal::

      46	 1.0836364e+00	 1.6253021e-01	 1.1333911e+00	 1.9102324e-01	[ 1.0556676e+00]	 2.1484709e-01


.. parsed-literal::

      47	 1.0969466e+00	 1.5899559e-01	 1.1470730e+00	 1.8852096e-01	[ 1.0656934e+00]	 2.0154476e-01


.. parsed-literal::

      48	 1.1115510e+00	 1.5913736e-01	 1.1619844e+00	 1.8941060e-01	[ 1.0771732e+00]	 2.1848607e-01
      49	 1.1223035e+00	 1.5731276e-01	 1.1733253e+00	 1.8826375e-01	[ 1.0844567e+00]	 1.8220377e-01


.. parsed-literal::

      50	 1.1382361e+00	 1.5405747e-01	 1.1905774e+00	 1.8638502e-01	[ 1.0895690e+00]	 2.1401834e-01


.. parsed-literal::

      51	 1.1484297e+00	 1.5202683e-01	 1.2014213e+00	 1.8443890e-01	[ 1.0949886e+00]	 2.1264458e-01


.. parsed-literal::

      52	 1.1593836e+00	 1.5079740e-01	 1.2123307e+00	 1.8442577e-01	[ 1.0980547e+00]	 2.3384023e-01


.. parsed-literal::

      53	 1.1679371e+00	 1.4974184e-01	 1.2210516e+00	 1.8323009e-01	[ 1.1025741e+00]	 2.1631384e-01
      54	 1.1792711e+00	 1.4789511e-01	 1.2325932e+00	 1.8236797e-01	[ 1.1052355e+00]	 1.9688225e-01


.. parsed-literal::

      55	 1.1933814e+00	 1.4651591e-01	 1.2465893e+00	 1.8250132e-01	[ 1.1147222e+00]	 2.0251846e-01


.. parsed-literal::

      56	 1.2074872e+00	 1.4490649e-01	 1.2605759e+00	 1.8258361e-01	[ 1.1218724e+00]	 2.1150279e-01


.. parsed-literal::

      57	 1.2198807e+00	 1.4404848e-01	 1.2734477e+00	 1.8226350e-01	[ 1.1279096e+00]	 2.1713686e-01


.. parsed-literal::

      58	 1.2322271e+00	 1.4427369e-01	 1.2856457e+00	 1.8250927e-01	[ 1.1410744e+00]	 2.0871258e-01


.. parsed-literal::

      59	 1.2436357e+00	 1.4378458e-01	 1.2977858e+00	 1.8191356e-01	  1.1394633e+00 	 2.1139026e-01


.. parsed-literal::

      60	 1.2545421e+00	 1.4328085e-01	 1.3090997e+00	 1.8133192e-01	[ 1.1411126e+00]	 2.0270085e-01


.. parsed-literal::

      61	 1.2643311e+00	 1.4222388e-01	 1.3190538e+00	 1.7917295e-01	[ 1.1422676e+00]	 2.1227694e-01


.. parsed-literal::

      62	 1.2726509e+00	 1.4160107e-01	 1.3273222e+00	 1.7852620e-01	[ 1.1548639e+00]	 2.1621466e-01


.. parsed-literal::

      63	 1.2800623e+00	 1.4128882e-01	 1.3349425e+00	 1.7784062e-01	[ 1.1612291e+00]	 2.0845294e-01


.. parsed-literal::

      64	 1.2872751e+00	 1.4069571e-01	 1.3424586e+00	 1.7701017e-01	[ 1.1703543e+00]	 2.0967436e-01


.. parsed-literal::

      65	 1.2950654e+00	 1.4039020e-01	 1.3505621e+00	 1.7564121e-01	[ 1.1787115e+00]	 2.0732355e-01


.. parsed-literal::

      66	 1.3033503e+00	 1.3974993e-01	 1.3589673e+00	 1.7515229e-01	[ 1.1802585e+00]	 2.0801902e-01


.. parsed-literal::

      67	 1.3134032e+00	 1.3886632e-01	 1.3694038e+00	 1.7444600e-01	  1.1755295e+00 	 2.0276380e-01


.. parsed-literal::

      68	 1.3222237e+00	 1.3755873e-01	 1.3786014e+00	 1.7388739e-01	  1.1691860e+00 	 2.0219636e-01
      69	 1.3308908e+00	 1.3711124e-01	 1.3869692e+00	 1.7350895e-01	  1.1745136e+00 	 2.0214915e-01


.. parsed-literal::

      70	 1.3361834e+00	 1.3650473e-01	 1.3921909e+00	 1.7305297e-01	  1.1745113e+00 	 1.7352939e-01
      71	 1.3425164e+00	 1.3555093e-01	 1.3987649e+00	 1.7207565e-01	  1.1739856e+00 	 1.7040515e-01


.. parsed-literal::

      72	 1.3491208e+00	 1.3474617e-01	 1.4055178e+00	 1.7173413e-01	  1.1724044e+00 	 1.9428015e-01
      73	 1.3552227e+00	 1.3425667e-01	 1.4116260e+00	 1.7153471e-01	  1.1746111e+00 	 1.6566920e-01


.. parsed-literal::

      74	 1.3622055e+00	 1.3380897e-01	 1.4186109e+00	 1.7133938e-01	[ 1.1829401e+00]	 1.7416549e-01
      75	 1.3666186e+00	 1.3337091e-01	 1.4233006e+00	 1.7132433e-01	  1.1786558e+00 	 1.8822789e-01


.. parsed-literal::

      76	 1.3714724e+00	 1.3319340e-01	 1.4281470e+00	 1.7128179e-01	[ 1.1864229e+00]	 2.1303630e-01


.. parsed-literal::

      77	 1.3758253e+00	 1.3298518e-01	 1.4326366e+00	 1.7115888e-01	[ 1.1904709e+00]	 2.1487188e-01


.. parsed-literal::

      78	 1.3808216e+00	 1.3264059e-01	 1.4378024e+00	 1.7102118e-01	[ 1.1911282e+00]	 2.1101284e-01


.. parsed-literal::

      79	 1.3869866e+00	 1.3222986e-01	 1.4441821e+00	 1.7023825e-01	[ 1.1950829e+00]	 2.0687366e-01


.. parsed-literal::

      80	 1.3923931e+00	 1.3143797e-01	 1.4496183e+00	 1.6959482e-01	[ 1.1957330e+00]	 2.0581436e-01


.. parsed-literal::

      81	 1.3963996e+00	 1.3115859e-01	 1.4534292e+00	 1.6927146e-01	[ 1.1975837e+00]	 2.1233296e-01
      82	 1.4012793e+00	 1.3069621e-01	 1.4581304e+00	 1.6853781e-01	[ 1.2005262e+00]	 2.0087171e-01


.. parsed-literal::

      83	 1.4035669e+00	 1.3019560e-01	 1.4605939e+00	 1.6751831e-01	  1.2000351e+00 	 1.9854140e-01
      84	 1.4077795e+00	 1.2999253e-01	 1.4647305e+00	 1.6736615e-01	[ 1.2042555e+00]	 1.7267132e-01


.. parsed-literal::

      85	 1.4118353e+00	 1.2970371e-01	 1.4688869e+00	 1.6708187e-01	[ 1.2077197e+00]	 2.0384216e-01


.. parsed-literal::

      86	 1.4149372e+00	 1.2941526e-01	 1.4720708e+00	 1.6670318e-01	[ 1.2103732e+00]	 2.0985699e-01
      87	 1.4229670e+00	 1.2847818e-01	 1.4801768e+00	 1.6546237e-01	[ 1.2189343e+00]	 1.8636012e-01


.. parsed-literal::

      88	 1.4240238e+00	 1.2740192e-01	 1.4818301e+00	 1.6326670e-01	  1.2108017e+00 	 1.7343354e-01
      89	 1.4323065e+00	 1.2709360e-01	 1.4897151e+00	 1.6325636e-01	[ 1.2248879e+00]	 1.8462348e-01


.. parsed-literal::

      90	 1.4344757e+00	 1.2699394e-01	 1.4918095e+00	 1.6328483e-01	[ 1.2272917e+00]	 1.8640780e-01


.. parsed-literal::

      91	 1.4377863e+00	 1.2651897e-01	 1.4953415e+00	 1.6267820e-01	  1.2260728e+00 	 2.0932722e-01
      92	 1.4411251e+00	 1.2613670e-01	 1.4987987e+00	 1.6201839e-01	  1.2265915e+00 	 1.9462419e-01


.. parsed-literal::

      93	 1.4446947e+00	 1.2577738e-01	 1.5024814e+00	 1.6137139e-01	  1.2254463e+00 	 1.9950294e-01
      94	 1.4485959e+00	 1.2525248e-01	 1.5065324e+00	 1.6041319e-01	  1.2208988e+00 	 1.8314123e-01


.. parsed-literal::

      95	 1.4518040e+00	 1.2505285e-01	 1.5096877e+00	 1.6012010e-01	  1.2183241e+00 	 2.0012379e-01
      96	 1.4548362e+00	 1.2483194e-01	 1.5126363e+00	 1.6004493e-01	  1.2178862e+00 	 2.0111179e-01


.. parsed-literal::

      97	 1.4588053e+00	 1.2469124e-01	 1.5167328e+00	 1.5959137e-01	  1.2143942e+00 	 2.0062518e-01
      98	 1.4626962e+00	 1.2433475e-01	 1.5205494e+00	 1.5916729e-01	  1.2163686e+00 	 1.9664335e-01


.. parsed-literal::

      99	 1.4648289e+00	 1.2421341e-01	 1.5227019e+00	 1.5887298e-01	  1.2176216e+00 	 2.0023584e-01
     100	 1.4686240e+00	 1.2403120e-01	 1.5266073e+00	 1.5841783e-01	  1.2206033e+00 	 2.0299149e-01


.. parsed-literal::

     101	 1.4717970e+00	 1.2372424e-01	 1.5299714e+00	 1.5787500e-01	  1.2177565e+00 	 1.9879508e-01
     102	 1.4753922e+00	 1.2363681e-01	 1.5336375e+00	 1.5778165e-01	  1.2209383e+00 	 1.9856906e-01


.. parsed-literal::

     103	 1.4778006e+00	 1.2356087e-01	 1.5360351e+00	 1.5780164e-01	  1.2206999e+00 	 1.9062352e-01


.. parsed-literal::

     104	 1.4805474e+00	 1.2355547e-01	 1.5388562e+00	 1.5819637e-01	  1.2161560e+00 	 2.0404482e-01


.. parsed-literal::

     105	 1.4832231e+00	 1.2349145e-01	 1.5416424e+00	 1.5814112e-01	  1.2050326e+00 	 2.1003342e-01
     106	 1.4856235e+00	 1.2344883e-01	 1.5440676e+00	 1.5818179e-01	  1.1984994e+00 	 1.9802785e-01


.. parsed-literal::

     107	 1.4898899e+00	 1.2329414e-01	 1.5484807e+00	 1.5807640e-01	  1.1802550e+00 	 1.9845939e-01


.. parsed-literal::

     108	 1.4905332e+00	 1.2319715e-01	 1.5492667e+00	 1.5812194e-01	  1.1615275e+00 	 2.0225883e-01
     109	 1.4935530e+00	 1.2308023e-01	 1.5521260e+00	 1.5790540e-01	  1.1728811e+00 	 1.9831634e-01


.. parsed-literal::

     110	 1.4952247e+00	 1.2296061e-01	 1.5537689e+00	 1.5770983e-01	  1.1748051e+00 	 2.0443583e-01


.. parsed-literal::

     111	 1.4975523e+00	 1.2274627e-01	 1.5560780e+00	 1.5739799e-01	  1.1742625e+00 	 2.0406747e-01


.. parsed-literal::

     112	 1.4997262e+00	 1.2254965e-01	 1.5583548e+00	 1.5707124e-01	  1.1638470e+00 	 2.0708919e-01


.. parsed-literal::

     113	 1.5024562e+00	 1.2241300e-01	 1.5610469e+00	 1.5693101e-01	  1.1593564e+00 	 2.1039629e-01


.. parsed-literal::

     114	 1.5044327e+00	 1.2235605e-01	 1.5630464e+00	 1.5696654e-01	  1.1515658e+00 	 2.0605040e-01


.. parsed-literal::

     115	 1.5062665e+00	 1.2236128e-01	 1.5649524e+00	 1.5706926e-01	  1.1413828e+00 	 2.0931530e-01
     116	 1.5078753e+00	 1.2204684e-01	 1.5667696e+00	 1.5688444e-01	  1.1268045e+00 	 2.0158148e-01


.. parsed-literal::

     117	 1.5105381e+00	 1.2212932e-01	 1.5694258e+00	 1.5706583e-01	  1.1269356e+00 	 2.0313358e-01


.. parsed-literal::

     118	 1.5121487e+00	 1.2206690e-01	 1.5710324e+00	 1.5703502e-01	  1.1340477e+00 	 2.0401144e-01


.. parsed-literal::

     119	 1.5136782e+00	 1.2192875e-01	 1.5726201e+00	 1.5701818e-01	  1.1385996e+00 	 2.0435405e-01
     120	 1.5155356e+00	 1.2171311e-01	 1.5746228e+00	 1.5685239e-01	  1.1423837e+00 	 1.9820404e-01


.. parsed-literal::

     121	 1.5176114e+00	 1.2159430e-01	 1.5767291e+00	 1.5698167e-01	  1.1409332e+00 	 2.0273972e-01


.. parsed-literal::

     122	 1.5192256e+00	 1.2152979e-01	 1.5784074e+00	 1.5707982e-01	  1.1338001e+00 	 2.0248699e-01
     123	 1.5208484e+00	 1.2147723e-01	 1.5801077e+00	 1.5712622e-01	  1.1244023e+00 	 2.0404148e-01


.. parsed-literal::

     124	 1.5235943e+00	 1.2139581e-01	 1.5829939e+00	 1.5720789e-01	  1.1050315e+00 	 1.9432235e-01
     125	 1.5242299e+00	 1.2117382e-01	 1.5839810e+00	 1.5707042e-01	  1.0708158e+00 	 1.8847704e-01


.. parsed-literal::

     126	 1.5272836e+00	 1.2114720e-01	 1.5867761e+00	 1.5702639e-01	  1.0832970e+00 	 2.0829773e-01


.. parsed-literal::

     127	 1.5283155e+00	 1.2107716e-01	 1.5877596e+00	 1.5696580e-01	  1.0858813e+00 	 2.0324469e-01
     128	 1.5302004e+00	 1.2090991e-01	 1.5896390e+00	 1.5684002e-01	  1.0841885e+00 	 1.9987226e-01


.. parsed-literal::

     129	 1.5325528e+00	 1.2063361e-01	 1.5920413e+00	 1.5664095e-01	  1.0757683e+00 	 1.8929076e-01


.. parsed-literal::

     130	 1.5341233e+00	 1.2033818e-01	 1.5938968e+00	 1.5637813e-01	  1.0596310e+00 	 2.1634579e-01


.. parsed-literal::

     131	 1.5362449e+00	 1.2029389e-01	 1.5959002e+00	 1.5631322e-01	  1.0607417e+00 	 2.0562029e-01


.. parsed-literal::

     132	 1.5372586e+00	 1.2026818e-01	 1.5969576e+00	 1.5634071e-01	  1.0546659e+00 	 2.0261979e-01
     133	 1.5384029e+00	 1.2008007e-01	 1.5982037e+00	 1.5609463e-01	  1.0445897e+00 	 2.0625925e-01


.. parsed-literal::

     134	 1.5399917e+00	 1.1995416e-01	 1.5998416e+00	 1.5607076e-01	  1.0405734e+00 	 1.8556738e-01


.. parsed-literal::

     135	 1.5416666e+00	 1.1971763e-01	 1.6015560e+00	 1.5592377e-01	  1.0300036e+00 	 2.1209669e-01
     136	 1.5428053e+00	 1.1950625e-01	 1.6026964e+00	 1.5578310e-01	  1.0281500e+00 	 2.0150042e-01


.. parsed-literal::

     137	 1.5440946e+00	 1.1938740e-01	 1.6039221e+00	 1.5565875e-01	  1.0256593e+00 	 2.0445800e-01
     138	 1.5457026e+00	 1.1926058e-01	 1.6054709e+00	 1.5547561e-01	  1.0252075e+00 	 2.0288992e-01


.. parsed-literal::

     139	 1.5469010e+00	 1.1916074e-01	 1.6067214e+00	 1.5518841e-01	  1.0263177e+00 	 1.9498301e-01


.. parsed-literal::

     140	 1.5482398e+00	 1.1909617e-01	 1.6080440e+00	 1.5507205e-01	  1.0244272e+00 	 2.0675755e-01
     141	 1.5495509e+00	 1.1902649e-01	 1.6093895e+00	 1.5493095e-01	  1.0202628e+00 	 1.8117690e-01


.. parsed-literal::

     142	 1.5507577e+00	 1.1893772e-01	 1.6106439e+00	 1.5478983e-01	  1.0144054e+00 	 2.0301843e-01
     143	 1.5513365e+00	 1.1881076e-01	 1.6113827e+00	 1.5450174e-01	  1.0090006e+00 	 1.7753315e-01


.. parsed-literal::

     144	 1.5529355e+00	 1.1875103e-01	 1.6129130e+00	 1.5447474e-01	  1.0059530e+00 	 2.1492362e-01
     145	 1.5535579e+00	 1.1870059e-01	 1.6135091e+00	 1.5442961e-01	  1.0056733e+00 	 1.9939733e-01


.. parsed-literal::

     146	 1.5547195e+00	 1.1859773e-01	 1.6146740e+00	 1.5426014e-01	  1.0035546e+00 	 2.0123839e-01


.. parsed-literal::

     147	 1.5553916e+00	 1.1850914e-01	 1.6153893e+00	 1.5401473e-01	  9.9719848e-01 	 3.1479812e-01
     148	 1.5564851e+00	 1.1846937e-01	 1.6164864e+00	 1.5388553e-01	  9.9464014e-01 	 1.9822383e-01


.. parsed-literal::

     149	 1.5576616e+00	 1.1844386e-01	 1.6177132e+00	 1.5374409e-01	  9.8585624e-01 	 2.1546841e-01


.. parsed-literal::

     150	 1.5589426e+00	 1.1843105e-01	 1.6190698e+00	 1.5366125e-01	  9.7152974e-01 	 2.1059680e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8780bc1e10>



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
    CPU times: user 974 ms, sys: 40 ms, total: 1.01 s
    Wall time: 380 ms


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

