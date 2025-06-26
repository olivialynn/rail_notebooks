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

    <pzflow.flow.Flow at 0x7f687fb40a30>



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
    0      23.994413  0.169586  0.102877  
    1      25.391064  0.152436  0.116988  
    2      24.304707  0.044925  0.023629  
    3      25.291103  0.044988  0.022629  
    4      25.096743  0.072897  0.038195  
    ...          ...       ...       ...  
    99995  24.737946  0.050621  0.034399  
    99996  24.224169  0.048735  0.047757  
    99997  25.613836  0.130475  0.116484  
    99998  25.274899  0.050757  0.028506  
    99999  25.699642  0.066114  0.050298  
    
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
          <td>27.490506</td>
          <td>0.770228</td>
          <td>26.677404</td>
          <td>0.161254</td>
          <td>26.159948</td>
          <td>0.090780</td>
          <td>25.206634</td>
          <td>0.063770</td>
          <td>24.597110</td>
          <td>0.071143</td>
          <td>23.868021</td>
          <td>0.084153</td>
          <td>0.169586</td>
          <td>0.102877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.149149</td>
          <td>0.610289</td>
          <td>27.314353</td>
          <td>0.274415</td>
          <td>26.710995</td>
          <td>0.146646</td>
          <td>26.175743</td>
          <td>0.148973</td>
          <td>25.715424</td>
          <td>0.187910</td>
          <td>25.487192</td>
          <td>0.331306</td>
          <td>0.152436</td>
          <td>0.116988</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.668615</td>
          <td>0.324973</td>
          <td>26.062800</td>
          <td>0.135164</td>
          <td>25.020566</td>
          <td>0.103288</td>
          <td>24.352314</td>
          <td>0.128519</td>
          <td>0.044925</td>
          <td>0.023629</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.985205</td>
          <td>0.463927</td>
          <td>27.657459</td>
          <td>0.322100</td>
          <td>26.673923</td>
          <td>0.226985</td>
          <td>25.648900</td>
          <td>0.177623</td>
          <td>24.876871</td>
          <td>0.201121</td>
          <td>0.044988</td>
          <td>0.022629</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.472767</td>
          <td>0.369122</td>
          <td>26.062249</td>
          <td>0.094633</td>
          <td>26.025579</td>
          <td>0.080647</td>
          <td>25.577974</td>
          <td>0.088539</td>
          <td>25.622137</td>
          <td>0.173634</td>
          <td>25.355266</td>
          <td>0.298160</td>
          <td>0.072897</td>
          <td>0.038195</td>
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
          <td>28.056176</td>
          <td>1.093766</td>
          <td>26.321871</td>
          <td>0.118707</td>
          <td>25.371633</td>
          <td>0.045170</td>
          <td>25.161166</td>
          <td>0.061250</td>
          <td>24.646244</td>
          <td>0.074303</td>
          <td>24.713307</td>
          <td>0.175178</td>
          <td>0.050621</td>
          <td>0.034399</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.554743</td>
          <td>0.803319</td>
          <td>26.816224</td>
          <td>0.181448</td>
          <td>26.238304</td>
          <td>0.097246</td>
          <td>25.181554</td>
          <td>0.062368</td>
          <td>24.741536</td>
          <td>0.080826</td>
          <td>24.229107</td>
          <td>0.115478</td>
          <td>0.048735</td>
          <td>0.047757</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.902553</td>
          <td>0.998829</td>
          <td>26.877970</td>
          <td>0.191160</td>
          <td>26.391909</td>
          <td>0.111231</td>
          <td>26.120800</td>
          <td>0.142097</td>
          <td>26.098049</td>
          <td>0.258361</td>
          <td>26.270107</td>
          <td>0.597377</td>
          <td>0.130475</td>
          <td>0.116484</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.271178</td>
          <td>0.314855</td>
          <td>26.429778</td>
          <td>0.130346</td>
          <td>26.070064</td>
          <td>0.083873</td>
          <td>26.224630</td>
          <td>0.155351</td>
          <td>25.573928</td>
          <td>0.166655</td>
          <td>25.358353</td>
          <td>0.298901</td>
          <td>0.050757</td>
          <td>0.028506</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.187637</td>
          <td>0.294466</td>
          <td>26.685513</td>
          <td>0.162374</td>
          <td>26.471396</td>
          <td>0.119204</td>
          <td>26.481667</td>
          <td>0.193267</td>
          <td>25.788066</td>
          <td>0.199765</td>
          <td>25.289891</td>
          <td>0.282829</td>
          <td>0.066114</td>
          <td>0.050298</td>
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
          <td>27.417355</td>
          <td>0.829993</td>
          <td>27.057238</td>
          <td>0.267692</td>
          <td>26.031736</td>
          <td>0.101429</td>
          <td>25.186632</td>
          <td>0.079202</td>
          <td>24.827627</td>
          <td>0.108984</td>
          <td>24.062014</td>
          <td>0.125460</td>
          <td>0.169586</td>
          <td>0.102877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.777807</td>
          <td>0.192196</td>
          <td>26.277383</td>
          <td>0.202683</td>
          <td>25.996154</td>
          <td>0.291545</td>
          <td>24.846478</td>
          <td>0.243196</td>
          <td>0.152436</td>
          <td>0.116988</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.101968</td>
          <td>1.093653</td>
          <td>28.546953</td>
          <td>0.714925</td>
          <td>25.975649</td>
          <td>0.148555</td>
          <td>24.958141</td>
          <td>0.115329</td>
          <td>24.385060</td>
          <td>0.156382</td>
          <td>0.044925</td>
          <td>0.023629</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.094880</td>
          <td>0.646674</td>
          <td>28.809545</td>
          <td>0.917353</td>
          <td>27.411114</td>
          <td>0.307581</td>
          <td>26.054727</td>
          <td>0.158957</td>
          <td>25.492950</td>
          <td>0.182594</td>
          <td>25.875277</td>
          <td>0.517218</td>
          <td>0.044988</td>
          <td>0.022629</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.743983</td>
          <td>0.505437</td>
          <td>26.210959</td>
          <td>0.125445</td>
          <td>25.937373</td>
          <td>0.088761</td>
          <td>25.500137</td>
          <td>0.099020</td>
          <td>25.184947</td>
          <td>0.141331</td>
          <td>25.455184</td>
          <td>0.378950</td>
          <td>0.072897</td>
          <td>0.038195</td>
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
          <td>27.080559</td>
          <td>0.641063</td>
          <td>26.135653</td>
          <td>0.116985</td>
          <td>25.543714</td>
          <td>0.062373</td>
          <td>25.089426</td>
          <td>0.068594</td>
          <td>24.698299</td>
          <td>0.092064</td>
          <td>24.637685</td>
          <td>0.194172</td>
          <td>0.050621</td>
          <td>0.034399</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.300878</td>
          <td>0.745426</td>
          <td>26.812478</td>
          <td>0.208821</td>
          <td>26.023271</td>
          <td>0.095383</td>
          <td>25.076199</td>
          <td>0.067904</td>
          <td>24.612167</td>
          <td>0.085480</td>
          <td>24.331817</td>
          <td>0.149934</td>
          <td>0.048735</td>
          <td>0.047757</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.087345</td>
          <td>0.661132</td>
          <td>26.485473</td>
          <td>0.164287</td>
          <td>26.265951</td>
          <td>0.122865</td>
          <td>26.137973</td>
          <td>0.178526</td>
          <td>25.642450</td>
          <td>0.216161</td>
          <td>25.830897</td>
          <td>0.520385</td>
          <td>0.130475</td>
          <td>0.116484</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.878539</td>
          <td>1.792201</td>
          <td>26.250657</td>
          <td>0.129180</td>
          <td>25.939365</td>
          <td>0.088416</td>
          <td>25.563216</td>
          <td>0.104035</td>
          <td>25.686390</td>
          <td>0.215120</td>
          <td>25.010692</td>
          <td>0.264450</td>
          <td>0.050757</td>
          <td>0.028506</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.371216</td>
          <td>0.381390</td>
          <td>27.070020</td>
          <td>0.259232</td>
          <td>26.581569</td>
          <td>0.155408</td>
          <td>26.498943</td>
          <td>0.232753</td>
          <td>27.079747</td>
          <td>0.635064</td>
          <td>25.233945</td>
          <td>0.318436</td>
          <td>0.066114</td>
          <td>0.050298</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.474473</td>
          <td>0.162085</td>
          <td>25.985363</td>
          <td>0.095675</td>
          <td>25.162502</td>
          <td>0.076099</td>
          <td>24.639100</td>
          <td>0.090773</td>
          <td>24.259140</td>
          <td>0.146068</td>
          <td>0.169586</td>
          <td>0.102877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.013469</td>
          <td>1.174344</td>
          <td>27.914261</td>
          <td>0.511312</td>
          <td>26.689122</td>
          <td>0.174801</td>
          <td>26.386136</td>
          <td>0.217516</td>
          <td>26.028214</td>
          <td>0.293578</td>
          <td>24.980273</td>
          <td>0.266085</td>
          <td>0.152436</td>
          <td>0.116988</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.849424</td>
          <td>0.380383</td>
          <td>25.928457</td>
          <td>0.122466</td>
          <td>25.066724</td>
          <td>0.109384</td>
          <td>24.280249</td>
          <td>0.122870</td>
          <td>0.044925</td>
          <td>0.023629</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.430218</td>
          <td>1.355043</td>
          <td>31.410059</td>
          <td>2.840283</td>
          <td>27.250627</td>
          <td>0.235076</td>
          <td>26.149084</td>
          <td>0.148136</td>
          <td>25.601529</td>
          <td>0.173416</td>
          <td>25.157811</td>
          <td>0.258129</td>
          <td>0.044988</td>
          <td>0.022629</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.109040</td>
          <td>0.284240</td>
          <td>26.057979</td>
          <td>0.097916</td>
          <td>25.783035</td>
          <td>0.067989</td>
          <td>25.799772</td>
          <td>0.112514</td>
          <td>25.687908</td>
          <td>0.191393</td>
          <td>25.165740</td>
          <td>0.266533</td>
          <td>0.072897</td>
          <td>0.038195</td>
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
          <td>26.600015</td>
          <td>0.413459</td>
          <td>26.268740</td>
          <td>0.115791</td>
          <td>25.449134</td>
          <td>0.049614</td>
          <td>25.154033</td>
          <td>0.062482</td>
          <td>24.814089</td>
          <td>0.088325</td>
          <td>24.811986</td>
          <td>0.195180</td>
          <td>0.050621</td>
          <td>0.034399</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.682632</td>
          <td>0.441845</td>
          <td>26.651612</td>
          <td>0.161886</td>
          <td>26.042393</td>
          <td>0.084411</td>
          <td>25.145880</td>
          <td>0.062422</td>
          <td>24.924561</td>
          <td>0.097902</td>
          <td>24.162597</td>
          <td>0.112472</td>
          <td>0.048735</td>
          <td>0.047757</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.854317</td>
          <td>0.547606</td>
          <td>26.687958</td>
          <td>0.188574</td>
          <td>26.366221</td>
          <td>0.129007</td>
          <td>26.232119</td>
          <td>0.186039</td>
          <td>25.935073</td>
          <td>0.265407</td>
          <td>25.663686</td>
          <td>0.444392</td>
          <td>0.130475</td>
          <td>0.116484</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.958606</td>
          <td>0.539401</td>
          <td>26.140535</td>
          <td>0.103334</td>
          <td>25.952816</td>
          <td>0.077350</td>
          <td>25.881899</td>
          <td>0.118252</td>
          <td>25.654326</td>
          <td>0.182327</td>
          <td>25.242588</td>
          <td>0.278079</td>
          <td>0.050757</td>
          <td>0.028506</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.393601</td>
          <td>0.356785</td>
          <td>27.233987</td>
          <td>0.266470</td>
          <td>26.561069</td>
          <td>0.134637</td>
          <td>26.154386</td>
          <td>0.153100</td>
          <td>25.750733</td>
          <td>0.202021</td>
          <td>25.609490</td>
          <td>0.380080</td>
          <td>0.066114</td>
          <td>0.050298</td>
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
