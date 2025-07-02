Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f5fcc898940>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.027399  0.018240  
    1      25.391064  0.106817  0.101895  
    2      24.304707  0.020341  0.013994  
    3      25.291103  0.022041  0.013166  
    4      25.096743  0.230967  0.156690  
    ...          ...       ...       ...  
    99995  24.737946  0.010600  0.009471  
    99996  24.224169  0.011249  0.009547  
    99997  25.613836  0.044481  0.033893  
    99998  25.274899  0.137212  0.109319  
    99999  25.699642  0.129982  0.126251  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.722221</td>
          <td>0.167533</td>
          <td>25.970965</td>
          <td>0.076851</td>
          <td>25.256917</td>
          <td>0.066677</td>
          <td>24.780110</td>
          <td>0.083623</td>
          <td>24.056442</td>
          <td>0.099309</td>
          <td>0.027399</td>
          <td>0.018240</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.041049</td>
          <td>0.565161</td>
          <td>26.950088</td>
          <td>0.203106</td>
          <td>26.297760</td>
          <td>0.102447</td>
          <td>26.498481</td>
          <td>0.196023</td>
          <td>25.984041</td>
          <td>0.235222</td>
          <td>25.046142</td>
          <td>0.231619</td>
          <td>0.106817</td>
          <td>0.101895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.719107</td>
          <td>1.400127</td>
          <td>28.103341</td>
          <td>0.455059</td>
          <td>25.819768</td>
          <td>0.109443</td>
          <td>24.986786</td>
          <td>0.100278</td>
          <td>24.417110</td>
          <td>0.135926</td>
          <td>0.020341</td>
          <td>0.013994</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.145094</td>
          <td>0.608548</td>
          <td>29.105197</td>
          <td>0.992661</td>
          <td>27.074726</td>
          <td>0.199793</td>
          <td>26.231469</td>
          <td>0.156263</td>
          <td>25.761969</td>
          <td>0.195429</td>
          <td>25.435655</td>
          <td>0.317998</td>
          <td>0.022041</td>
          <td>0.013166</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.830358</td>
          <td>0.219806</td>
          <td>26.020802</td>
          <td>0.091254</td>
          <td>25.893837</td>
          <td>0.071785</td>
          <td>25.736851</td>
          <td>0.101790</td>
          <td>25.504978</td>
          <td>0.157126</td>
          <td>25.149481</td>
          <td>0.252225</td>
          <td>0.230967</td>
          <td>0.156690</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>31.389703</td>
          <td>3.962163</td>
          <td>26.538882</td>
          <td>0.143206</td>
          <td>25.341046</td>
          <td>0.043960</td>
          <td>25.147651</td>
          <td>0.060520</td>
          <td>24.944422</td>
          <td>0.096622</td>
          <td>24.794684</td>
          <td>0.187674</td>
          <td>0.010600</td>
          <td>0.009471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.035445</td>
          <td>1.080656</td>
          <td>27.013094</td>
          <td>0.214096</td>
          <td>26.113769</td>
          <td>0.087165</td>
          <td>25.211315</td>
          <td>0.064036</td>
          <td>24.823344</td>
          <td>0.086869</td>
          <td>24.279667</td>
          <td>0.120669</td>
          <td>0.011249</td>
          <td>0.009547</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.682743</td>
          <td>0.433808</td>
          <td>26.732471</td>
          <td>0.169001</td>
          <td>26.350544</td>
          <td>0.107286</td>
          <td>26.317875</td>
          <td>0.168228</td>
          <td>25.901743</td>
          <td>0.219691</td>
          <td>25.644574</td>
          <td>0.374925</td>
          <td>0.044481</td>
          <td>0.033893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.171057</td>
          <td>0.290560</td>
          <td>26.222399</td>
          <td>0.108861</td>
          <td>25.890572</td>
          <td>0.071578</td>
          <td>25.670596</td>
          <td>0.096047</td>
          <td>25.558189</td>
          <td>0.164434</td>
          <td>25.380847</td>
          <td>0.304353</td>
          <td>0.137212</td>
          <td>0.109319</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.348147</td>
          <td>0.334709</td>
          <td>26.809689</td>
          <td>0.180447</td>
          <td>26.641506</td>
          <td>0.138128</td>
          <td>26.081095</td>
          <td>0.137316</td>
          <td>26.282200</td>
          <td>0.300009</td>
          <td>25.284478</td>
          <td>0.281591</td>
          <td>0.129982</td>
          <td>0.126251</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.300151</td>
          <td>1.349028</td>
          <td>26.683095</td>
          <td>0.186339</td>
          <td>26.076024</td>
          <td>0.099298</td>
          <td>25.130445</td>
          <td>0.070796</td>
          <td>24.713353</td>
          <td>0.092871</td>
          <td>23.955057</td>
          <td>0.107531</td>
          <td>0.027399</td>
          <td>0.018240</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.541467</td>
          <td>1.548338</td>
          <td>27.952732</td>
          <td>0.524846</td>
          <td>26.545551</td>
          <td>0.154280</td>
          <td>26.244063</td>
          <td>0.192635</td>
          <td>25.945968</td>
          <td>0.274029</td>
          <td>25.071975</td>
          <td>0.285996</td>
          <td>0.106817</td>
          <td>0.101895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.711557</td>
          <td>0.388595</td>
          <td>26.074004</td>
          <td>0.161068</td>
          <td>25.256752</td>
          <td>0.148821</td>
          <td>24.336044</td>
          <td>0.149450</td>
          <td>0.020341</td>
          <td>0.013994</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.122637</td>
          <td>3.833063</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.261535</td>
          <td>0.271781</td>
          <td>26.626748</td>
          <td>0.255978</td>
          <td>26.105992</td>
          <td>0.302094</td>
          <td>25.952272</td>
          <td>0.545564</td>
          <td>0.022041</td>
          <td>0.013166</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.935094</td>
          <td>0.619923</td>
          <td>26.054156</td>
          <td>0.120308</td>
          <td>25.872596</td>
          <td>0.093078</td>
          <td>25.808613</td>
          <td>0.143972</td>
          <td>25.545038</td>
          <td>0.212197</td>
          <td>25.127701</td>
          <td>0.322244</td>
          <td>0.230967</td>
          <td>0.156690</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.351131</td>
          <td>0.767357</td>
          <td>26.250189</td>
          <td>0.128510</td>
          <td>25.433643</td>
          <td>0.056231</td>
          <td>25.025374</td>
          <td>0.064406</td>
          <td>24.925077</td>
          <td>0.111608</td>
          <td>24.524354</td>
          <td>0.175390</td>
          <td>0.010600</td>
          <td>0.009471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.053194</td>
          <td>0.626693</td>
          <td>26.986438</td>
          <td>0.239744</td>
          <td>26.078338</td>
          <td>0.099354</td>
          <td>25.296899</td>
          <td>0.081881</td>
          <td>24.717016</td>
          <td>0.093033</td>
          <td>24.178229</td>
          <td>0.130358</td>
          <td>0.011249</td>
          <td>0.009547</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.567757</td>
          <td>0.441462</td>
          <td>26.788592</td>
          <td>0.204236</td>
          <td>26.340412</td>
          <td>0.125467</td>
          <td>26.726199</td>
          <td>0.278722</td>
          <td>27.094385</td>
          <td>0.638375</td>
          <td>25.823091</td>
          <td>0.498175</td>
          <td>0.044481</td>
          <td>0.033893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.437853</td>
          <td>0.411938</td>
          <td>26.349604</td>
          <td>0.146288</td>
          <td>26.151012</td>
          <td>0.111196</td>
          <td>25.784271</td>
          <td>0.131890</td>
          <td>25.654507</td>
          <td>0.218385</td>
          <td>26.150986</td>
          <td>0.653664</td>
          <td>0.137212</td>
          <td>0.109319</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.016475</td>
          <td>0.296563</td>
          <td>26.544645</td>
          <td>0.173294</td>
          <td>26.514912</td>
          <td>0.152818</td>
          <td>26.938302</td>
          <td>0.345410</td>
          <td>25.645944</td>
          <td>0.217504</td>
          <td>25.944251</td>
          <td>0.566572</td>
          <td>0.129982</td>
          <td>0.126251</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.597198</td>
          <td>0.828778</td>
          <td>26.601716</td>
          <td>0.152082</td>
          <td>26.036059</td>
          <td>0.081995</td>
          <td>25.254470</td>
          <td>0.067049</td>
          <td>24.673253</td>
          <td>0.076659</td>
          <td>24.049508</td>
          <td>0.099458</td>
          <td>0.027399</td>
          <td>0.018240</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.735876</td>
          <td>0.488634</td>
          <td>27.227147</td>
          <td>0.283960</td>
          <td>26.804172</td>
          <td>0.179985</td>
          <td>26.340035</td>
          <td>0.195166</td>
          <td>26.066174</td>
          <td>0.283662</td>
          <td>25.094468</td>
          <td>0.272992</td>
          <td>0.106817</td>
          <td>0.101895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.334114</td>
          <td>0.599982</td>
          <td>28.232763</td>
          <td>0.502844</td>
          <td>26.151615</td>
          <td>0.146536</td>
          <td>25.085919</td>
          <td>0.109807</td>
          <td>24.370579</td>
          <td>0.131120</td>
          <td>0.020341</td>
          <td>0.013994</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.851719</td>
          <td>1.501083</td>
          <td>27.258464</td>
          <td>0.233865</td>
          <td>26.005789</td>
          <td>0.129255</td>
          <td>25.532645</td>
          <td>0.161585</td>
          <td>24.712202</td>
          <td>0.175798</td>
          <td>0.022041</td>
          <td>0.013166</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.197125</td>
          <td>0.376921</td>
          <td>26.016826</td>
          <td>0.124244</td>
          <td>26.043443</td>
          <td>0.115875</td>
          <td>25.597048</td>
          <td>0.128694</td>
          <td>25.459760</td>
          <td>0.211197</td>
          <td>25.265565</td>
          <td>0.382760</td>
          <td>0.230967</td>
          <td>0.156690</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.311408</td>
          <td>0.117772</td>
          <td>25.446366</td>
          <td>0.048335</td>
          <td>25.128334</td>
          <td>0.059579</td>
          <td>24.802689</td>
          <td>0.085421</td>
          <td>24.620351</td>
          <td>0.162071</td>
          <td>0.010600</td>
          <td>0.009471</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.642267</td>
          <td>1.501171</td>
          <td>26.729168</td>
          <td>0.168736</td>
          <td>26.087254</td>
          <td>0.085280</td>
          <td>25.237194</td>
          <td>0.065624</td>
          <td>24.640210</td>
          <td>0.074018</td>
          <td>24.176041</td>
          <td>0.110426</td>
          <td>0.011249</td>
          <td>0.009547</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.961135</td>
          <td>0.539883</td>
          <td>27.300565</td>
          <td>0.275925</td>
          <td>26.433434</td>
          <td>0.117729</td>
          <td>26.138752</td>
          <td>0.147421</td>
          <td>25.429838</td>
          <td>0.150340</td>
          <td>25.548630</td>
          <td>0.354567</td>
          <td>0.044481</td>
          <td>0.033893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.556008</td>
          <td>0.439239</td>
          <td>26.180035</td>
          <td>0.122087</td>
          <td>26.040975</td>
          <td>0.097168</td>
          <td>25.941707</td>
          <td>0.145234</td>
          <td>26.146061</td>
          <td>0.314732</td>
          <td>25.343288</td>
          <td>0.347024</td>
          <td>0.137212</td>
          <td>0.109319</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.235463</td>
          <td>0.719216</td>
          <td>26.603014</td>
          <td>0.177154</td>
          <td>26.463265</td>
          <td>0.141774</td>
          <td>26.610115</td>
          <td>0.257548</td>
          <td>26.511270</td>
          <td>0.422651</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.129982</td>
          <td>0.126251</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
