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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fcd5d0dc7f0>



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
          <td>26.804965</td>
          <td>0.475546</td>
          <td>26.960133</td>
          <td>0.204823</td>
          <td>25.999443</td>
          <td>0.078808</td>
          <td>25.187253</td>
          <td>0.062684</td>
          <td>24.686743</td>
          <td>0.077010</td>
          <td>24.001783</td>
          <td>0.094660</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.137704</td>
          <td>0.237432</td>
          <td>26.933133</td>
          <td>0.177282</td>
          <td>26.281420</td>
          <td>0.163080</td>
          <td>26.228021</td>
          <td>0.287184</td>
          <td>25.092087</td>
          <td>0.240586</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.728991</td>
          <td>1.304879</td>
          <td>26.043285</td>
          <td>0.132904</td>
          <td>25.055136</td>
          <td>0.106458</td>
          <td>24.338396</td>
          <td>0.126978</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.075107</td>
          <td>1.105818</td>
          <td>28.397363</td>
          <td>0.625491</td>
          <td>26.935560</td>
          <td>0.177648</td>
          <td>26.014979</td>
          <td>0.129688</td>
          <td>25.585149</td>
          <td>0.168256</td>
          <td>25.388613</td>
          <td>0.306255</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.726969</td>
          <td>0.448554</td>
          <td>26.020211</td>
          <td>0.091207</td>
          <td>25.955722</td>
          <td>0.075823</td>
          <td>25.534198</td>
          <td>0.085191</td>
          <td>25.452577</td>
          <td>0.150227</td>
          <td>25.345506</td>
          <td>0.295826</td>
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
          <td>26.797009</td>
          <td>0.472734</td>
          <td>26.325509</td>
          <td>0.119083</td>
          <td>25.456923</td>
          <td>0.048722</td>
          <td>25.022251</td>
          <td>0.054146</td>
          <td>24.820677</td>
          <td>0.086665</td>
          <td>24.491459</td>
          <td>0.144920</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.416363</td>
          <td>0.353192</td>
          <td>26.492516</td>
          <td>0.137601</td>
          <td>26.103900</td>
          <td>0.086411</td>
          <td>25.209802</td>
          <td>0.063950</td>
          <td>24.773229</td>
          <td>0.083117</td>
          <td>24.128877</td>
          <td>0.105809</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.254030</td>
          <td>1.987680</td>
          <td>26.747122</td>
          <td>0.171120</td>
          <td>26.361068</td>
          <td>0.108277</td>
          <td>25.928248</td>
          <td>0.120289</td>
          <td>26.017648</td>
          <td>0.241842</td>
          <td>25.100399</td>
          <td>0.242242</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.854407</td>
          <td>0.224239</td>
          <td>26.295211</td>
          <td>0.115988</td>
          <td>26.041016</td>
          <td>0.081753</td>
          <td>25.818555</td>
          <td>0.109327</td>
          <td>25.777748</td>
          <td>0.198040</td>
          <td>24.855435</td>
          <td>0.197532</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.509071</td>
          <td>0.379692</td>
          <td>26.703085</td>
          <td>0.164825</td>
          <td>26.601935</td>
          <td>0.133488</td>
          <td>26.095698</td>
          <td>0.139057</td>
          <td>26.237326</td>
          <td>0.289352</td>
          <td>25.708709</td>
          <td>0.394036</td>
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
          <td>26.731555</td>
          <td>0.497218</td>
          <td>26.899801</td>
          <td>0.223085</td>
          <td>26.255694</td>
          <td>0.115959</td>
          <td>25.164619</td>
          <td>0.072830</td>
          <td>24.617966</td>
          <td>0.085245</td>
          <td>23.899580</td>
          <td>0.102251</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.411583</td>
          <td>1.428414</td>
          <td>27.444066</td>
          <td>0.346823</td>
          <td>26.510737</td>
          <td>0.144628</td>
          <td>26.936827</td>
          <td>0.328506</td>
          <td>25.815896</td>
          <td>0.238296</td>
          <td>24.954859</td>
          <td>0.251306</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.536939</td>
          <td>0.875956</td>
          <td>29.729866</td>
          <td>1.545671</td>
          <td>27.557161</td>
          <td>0.351070</td>
          <td>26.080508</td>
          <td>0.165466</td>
          <td>24.948474</td>
          <td>0.116417</td>
          <td>24.199829</td>
          <td>0.135796</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.920391</td>
          <td>1.730003</td>
          <td>27.287182</td>
          <td>0.294897</td>
          <td>26.219865</td>
          <td>0.194702</td>
          <td>25.453988</td>
          <td>0.187688</td>
          <td>25.473516</td>
          <td>0.404190</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.619256</td>
          <td>0.457444</td>
          <td>26.131313</td>
          <td>0.115923</td>
          <td>25.974067</td>
          <td>0.090665</td>
          <td>25.510681</td>
          <td>0.098809</td>
          <td>25.286933</td>
          <td>0.152623</td>
          <td>24.593541</td>
          <td>0.185975</td>
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
          <td>27.117705</td>
          <td>0.663586</td>
          <td>26.352150</td>
          <td>0.142924</td>
          <td>25.521199</td>
          <td>0.062056</td>
          <td>25.131306</td>
          <td>0.072282</td>
          <td>24.900332</td>
          <td>0.111492</td>
          <td>24.472860</td>
          <td>0.171369</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.316120</td>
          <td>0.751433</td>
          <td>26.660865</td>
          <td>0.183247</td>
          <td>26.054942</td>
          <td>0.097711</td>
          <td>25.152769</td>
          <td>0.072386</td>
          <td>24.796578</td>
          <td>0.100138</td>
          <td>24.275904</td>
          <td>0.142373</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.965967</td>
          <td>0.276727</td>
          <td>26.430381</td>
          <td>0.151744</td>
          <td>26.212808</td>
          <td>0.113134</td>
          <td>26.324871</td>
          <td>0.201550</td>
          <td>25.780532</td>
          <td>0.234184</td>
          <td>25.795076</td>
          <td>0.491141</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.826907</td>
          <td>0.250365</td>
          <td>26.263945</td>
          <td>0.133699</td>
          <td>26.077448</td>
          <td>0.102402</td>
          <td>25.839302</td>
          <td>0.135777</td>
          <td>25.261261</td>
          <td>0.153896</td>
          <td>25.956290</td>
          <td>0.561304</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.301695</td>
          <td>0.360917</td>
          <td>26.745666</td>
          <td>0.197824</td>
          <td>26.522661</td>
          <td>0.147530</td>
          <td>26.171121</td>
          <td>0.176542</td>
          <td>25.847412</td>
          <td>0.246833</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>27.162318</td>
          <td>0.616010</td>
          <td>26.558246</td>
          <td>0.145626</td>
          <td>25.980754</td>
          <td>0.077528</td>
          <td>25.166341</td>
          <td>0.061541</td>
          <td>24.720940</td>
          <td>0.079381</td>
          <td>24.181476</td>
          <td>0.110797</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.902046</td>
          <td>0.998972</td>
          <td>27.692836</td>
          <td>0.371263</td>
          <td>26.725024</td>
          <td>0.148561</td>
          <td>26.493479</td>
          <td>0.195385</td>
          <td>26.103848</td>
          <td>0.259819</td>
          <td>25.238341</td>
          <td>0.271480</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.598405</td>
          <td>0.367364</td>
          <td>28.082037</td>
          <td>0.480158</td>
          <td>25.833428</td>
          <td>0.120558</td>
          <td>25.182751</td>
          <td>0.128995</td>
          <td>24.295570</td>
          <td>0.133014</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.505814</td>
          <td>1.410806</td>
          <td>27.319211</td>
          <td>0.301608</td>
          <td>26.278150</td>
          <td>0.203754</td>
          <td>25.490728</td>
          <td>0.192931</td>
          <td>25.273435</td>
          <td>0.344760</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.204634</td>
          <td>0.298787</td>
          <td>26.177594</td>
          <td>0.104814</td>
          <td>25.977840</td>
          <td>0.077429</td>
          <td>25.776855</td>
          <td>0.105573</td>
          <td>25.744571</td>
          <td>0.192849</td>
          <td>25.347650</td>
          <td>0.296738</td>
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
          <td>27.270737</td>
          <td>0.693707</td>
          <td>26.215962</td>
          <td>0.115923</td>
          <td>25.442786</td>
          <td>0.052110</td>
          <td>25.012731</td>
          <td>0.058360</td>
          <td>24.976211</td>
          <td>0.107452</td>
          <td>24.901045</td>
          <td>0.221781</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.798428</td>
          <td>0.477862</td>
          <td>26.655878</td>
          <td>0.160538</td>
          <td>26.012377</td>
          <td>0.081044</td>
          <td>25.285491</td>
          <td>0.069590</td>
          <td>24.976772</td>
          <td>0.101047</td>
          <td>24.520076</td>
          <td>0.151032</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.610204</td>
          <td>0.422955</td>
          <td>26.467597</td>
          <td>0.140445</td>
          <td>26.291134</td>
          <td>0.106940</td>
          <td>26.320888</td>
          <td>0.177229</td>
          <td>25.644997</td>
          <td>0.185572</td>
          <td>25.656976</td>
          <td>0.395856</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.520181</td>
          <td>0.411584</td>
          <td>26.149280</td>
          <td>0.112907</td>
          <td>25.944906</td>
          <td>0.084281</td>
          <td>25.921626</td>
          <td>0.134603</td>
          <td>26.030184</td>
          <td>0.271872</td>
          <td>24.762048</td>
          <td>0.204562</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.383122</td>
          <td>1.334148</td>
          <td>26.520271</td>
          <td>0.145691</td>
          <td>26.746424</td>
          <td>0.157023</td>
          <td>26.388672</td>
          <td>0.185819</td>
          <td>25.931682</td>
          <td>0.233643</td>
          <td>26.820242</td>
          <td>0.890102</td>
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
