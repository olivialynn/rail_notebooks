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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f09b8c5c9d0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>27.736257</td>
          <td>0.901934</td>
          <td>26.625269</td>
          <td>0.154225</td>
          <td>26.031212</td>
          <td>0.081049</td>
          <td>25.238006</td>
          <td>0.065569</td>
          <td>24.737275</td>
          <td>0.080523</td>
          <td>23.937396</td>
          <td>0.089453</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.655635</td>
          <td>0.857204</td>
          <td>27.416437</td>
          <td>0.298030</td>
          <td>26.822803</td>
          <td>0.161390</td>
          <td>26.047855</td>
          <td>0.133430</td>
          <td>25.853726</td>
          <td>0.211064</td>
          <td>25.149581</td>
          <td>0.252245</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.732805</td>
          <td>0.899989</td>
          <td>28.820893</td>
          <td>0.831416</td>
          <td>27.775454</td>
          <td>0.353613</td>
          <td>26.070267</td>
          <td>0.136038</td>
          <td>24.818540</td>
          <td>0.086502</td>
          <td>24.383880</td>
          <td>0.132078</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.527399</td>
          <td>0.684337</td>
          <td>27.703028</td>
          <td>0.333973</td>
          <td>26.341236</td>
          <td>0.171606</td>
          <td>25.395797</td>
          <td>0.143071</td>
          <td>25.841980</td>
          <td>0.436334</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.145555</td>
          <td>0.284639</td>
          <td>25.918960</td>
          <td>0.083440</td>
          <td>26.009443</td>
          <td>0.079507</td>
          <td>25.664847</td>
          <td>0.095563</td>
          <td>25.324463</td>
          <td>0.134535</td>
          <td>25.540744</td>
          <td>0.345637</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.755496</td>
          <td>1.586168</td>
          <td>26.269690</td>
          <td>0.113441</td>
          <td>25.477196</td>
          <td>0.049607</td>
          <td>25.102094</td>
          <td>0.058123</td>
          <td>24.835989</td>
          <td>0.087841</td>
          <td>24.728919</td>
          <td>0.177514</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.067575</td>
          <td>0.575992</td>
          <td>26.965132</td>
          <td>0.205683</td>
          <td>25.881371</td>
          <td>0.070997</td>
          <td>25.278961</td>
          <td>0.067992</td>
          <td>24.846176</td>
          <td>0.088632</td>
          <td>24.037547</td>
          <td>0.097678</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.095348</td>
          <td>0.587501</td>
          <td>26.798476</td>
          <td>0.178742</td>
          <td>26.199850</td>
          <td>0.094019</td>
          <td>26.312093</td>
          <td>0.167402</td>
          <td>25.745882</td>
          <td>0.192800</td>
          <td>25.645157</td>
          <td>0.375095</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.694854</td>
          <td>0.437806</td>
          <td>26.229211</td>
          <td>0.109510</td>
          <td>26.025061</td>
          <td>0.080610</td>
          <td>25.869561</td>
          <td>0.114300</td>
          <td>25.813849</td>
          <td>0.204134</td>
          <td>25.193286</td>
          <td>0.261442</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.720289</td>
          <td>0.446301</td>
          <td>26.954533</td>
          <td>0.203864</td>
          <td>26.540545</td>
          <td>0.126580</td>
          <td>26.192550</td>
          <td>0.151137</td>
          <td>26.221832</td>
          <td>0.285750</td>
          <td>25.936809</td>
          <td>0.468627</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.400993</td>
          <td>0.387264</td>
          <td>26.797799</td>
          <td>0.204887</td>
          <td>25.970174</td>
          <td>0.090326</td>
          <td>25.290098</td>
          <td>0.081363</td>
          <td>24.557856</td>
          <td>0.080847</td>
          <td>23.991667</td>
          <td>0.110816</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.565170</td>
          <td>0.880954</td>
          <td>28.002656</td>
          <td>0.529988</td>
          <td>26.493807</td>
          <td>0.142536</td>
          <td>25.935944</td>
          <td>0.142970</td>
          <td>25.817689</td>
          <td>0.238650</td>
          <td>25.412498</td>
          <td>0.362849</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.836025</td>
          <td>0.544330</td>
          <td>27.820108</td>
          <td>0.471050</td>
          <td>28.023376</td>
          <td>0.500992</td>
          <td>25.920387</td>
          <td>0.144263</td>
          <td>25.100332</td>
          <td>0.132804</td>
          <td>24.300703</td>
          <td>0.148116</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.547044</td>
          <td>0.809103</td>
          <td>27.237074</td>
          <td>0.283199</td>
          <td>26.362686</td>
          <td>0.219434</td>
          <td>25.187256</td>
          <td>0.149557</td>
          <td>26.354467</td>
          <td>0.760643</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.766564</td>
          <td>0.510301</td>
          <td>26.085043</td>
          <td>0.111349</td>
          <td>25.913678</td>
          <td>0.085975</td>
          <td>25.671826</td>
          <td>0.113749</td>
          <td>25.310946</td>
          <td>0.155794</td>
          <td>24.744617</td>
          <td>0.211151</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>27.307422</td>
          <td>0.754392</td>
          <td>26.430218</td>
          <td>0.152826</td>
          <td>25.345128</td>
          <td>0.053085</td>
          <td>25.024380</td>
          <td>0.065757</td>
          <td>24.910807</td>
          <td>0.112515</td>
          <td>24.901829</td>
          <td>0.245440</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.363158</td>
          <td>1.395683</td>
          <td>26.633740</td>
          <td>0.179089</td>
          <td>26.224548</td>
          <td>0.113324</td>
          <td>25.180802</td>
          <td>0.074202</td>
          <td>24.855635</td>
          <td>0.105448</td>
          <td>24.150478</td>
          <td>0.127759</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.433303</td>
          <td>0.815436</td>
          <td>26.614443</td>
          <td>0.177513</td>
          <td>26.207176</td>
          <td>0.112580</td>
          <td>26.075471</td>
          <td>0.163191</td>
          <td>25.684535</td>
          <td>0.216235</td>
          <td>25.333513</td>
          <td>0.344984</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.519955</td>
          <td>0.433185</td>
          <td>26.120315</td>
          <td>0.118064</td>
          <td>25.951158</td>
          <td>0.091669</td>
          <td>26.065670</td>
          <td>0.164888</td>
          <td>25.398207</td>
          <td>0.172970</td>
          <td>25.592544</td>
          <td>0.428820</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.305201</td>
          <td>0.361908</td>
          <td>26.540286</td>
          <td>0.166276</td>
          <td>26.663704</td>
          <td>0.166453</td>
          <td>26.508278</td>
          <td>0.234207</td>
          <td>25.849593</td>
          <td>0.247276</td>
          <td>26.296982</td>
          <td>0.699930</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>28.767365</td>
          <td>1.595372</td>
          <td>26.372755</td>
          <td>0.124081</td>
          <td>25.953976</td>
          <td>0.075716</td>
          <td>25.280052</td>
          <td>0.068067</td>
          <td>24.628504</td>
          <td>0.073156</td>
          <td>24.039953</td>
          <td>0.097897</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.433757</td>
          <td>1.348600</td>
          <td>27.568035</td>
          <td>0.336601</td>
          <td>26.534325</td>
          <td>0.126017</td>
          <td>26.467292</td>
          <td>0.191122</td>
          <td>25.884496</td>
          <td>0.216751</td>
          <td>25.288243</td>
          <td>0.282708</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.583763</td>
          <td>1.506320</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.086235</td>
          <td>0.481659</td>
          <td>26.214711</td>
          <td>0.167395</td>
          <td>25.143581</td>
          <td>0.124690</td>
          <td>24.480099</td>
          <td>0.155898</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.664801</td>
          <td>0.969870</td>
          <td>31.235386</td>
          <td>2.874318</td>
          <td>27.650211</td>
          <td>0.391555</td>
          <td>26.127322</td>
          <td>0.179428</td>
          <td>25.274244</td>
          <td>0.160561</td>
          <td>24.520988</td>
          <td>0.186182</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.577619</td>
          <td>0.400686</td>
          <td>26.213897</td>
          <td>0.108189</td>
          <td>26.120064</td>
          <td>0.087775</td>
          <td>25.733996</td>
          <td>0.101687</td>
          <td>25.424085</td>
          <td>0.146799</td>
          <td>25.190407</td>
          <td>0.261186</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.552440</td>
          <td>0.154982</td>
          <td>25.471689</td>
          <td>0.053464</td>
          <td>25.028324</td>
          <td>0.059172</td>
          <td>24.850213</td>
          <td>0.096231</td>
          <td>24.493627</td>
          <td>0.157228</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.668586</td>
          <td>0.162288</td>
          <td>25.992423</td>
          <td>0.079630</td>
          <td>25.253959</td>
          <td>0.067673</td>
          <td>24.921796</td>
          <td>0.096294</td>
          <td>24.091678</td>
          <td>0.104183</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.197476</td>
          <td>0.306297</td>
          <td>26.619348</td>
          <td>0.159964</td>
          <td>26.429215</td>
          <td>0.120613</td>
          <td>26.474551</td>
          <td>0.201767</td>
          <td>25.757055</td>
          <td>0.203930</td>
          <td>25.270813</td>
          <td>0.291805</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.970061</td>
          <td>0.266343</td>
          <td>26.182042</td>
          <td>0.116171</td>
          <td>26.042409</td>
          <td>0.091831</td>
          <td>25.811433</td>
          <td>0.122351</td>
          <td>25.508721</td>
          <td>0.176150</td>
          <td>24.762737</td>
          <td>0.204680</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.921499</td>
          <td>0.204818</td>
          <td>26.705462</td>
          <td>0.151608</td>
          <td>26.175468</td>
          <td>0.154987</td>
          <td>25.901689</td>
          <td>0.227908</td>
          <td>25.401945</td>
          <td>0.321113</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
