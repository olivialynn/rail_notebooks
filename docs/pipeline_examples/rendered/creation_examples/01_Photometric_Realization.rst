Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
```Photometric_Realization.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fcefcd19ed0>



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
    0      23.994413  0.001189  0.000630  
    1      25.391064  0.054283  0.030536  
    2      24.304707  0.081860  0.064210  
    3      25.291103  0.046510  0.044264  
    4      25.096743  0.162684  0.154785  
    ...          ...       ...       ...  
    99995  24.737946  0.043542  0.042334  
    99996  24.224169  0.047388  0.026429  
    99997  25.613836  0.085606  0.067795  
    99998  25.274899  0.036149  0.031625  
    99999  25.699642  0.062642  0.037997  
    
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

    Inserting handle into data store.  output_truth: None, error_model
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
          <td>26.589610</td>
          <td>0.404048</td>
          <td>26.470193</td>
          <td>0.134977</td>
          <td>26.009399</td>
          <td>0.079503</td>
          <td>25.162245</td>
          <td>0.061309</td>
          <td>24.579678</td>
          <td>0.070054</td>
          <td>23.845345</td>
          <td>0.082488</td>
          <td>0.001189</td>
          <td>0.000630</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.563928</td>
          <td>0.335262</td>
          <td>26.673991</td>
          <td>0.142050</td>
          <td>26.110227</td>
          <td>0.140809</td>
          <td>25.748222</td>
          <td>0.193180</td>
          <td>25.418087</td>
          <td>0.313567</td>
          <td>0.054283</td>
          <td>0.030536</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.441787</td>
          <td>3.052948</td>
          <td>28.931877</td>
          <td>0.892185</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.011966</td>
          <td>0.129351</td>
          <td>24.954415</td>
          <td>0.097473</td>
          <td>24.145495</td>
          <td>0.107356</td>
          <td>0.081860</td>
          <td>0.064210</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.752757</td>
          <td>1.424611</td>
          <td>27.121275</td>
          <td>0.207747</td>
          <td>26.208466</td>
          <td>0.153214</td>
          <td>25.455516</td>
          <td>0.150606</td>
          <td>25.179702</td>
          <td>0.258552</td>
          <td>0.046510</td>
          <td>0.044264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.644301</td>
          <td>0.421313</td>
          <td>25.953447</td>
          <td>0.086010</td>
          <td>25.944525</td>
          <td>0.075076</td>
          <td>25.531787</td>
          <td>0.085010</td>
          <td>25.558807</td>
          <td>0.164521</td>
          <td>25.169421</td>
          <td>0.256384</td>
          <td>0.162684</td>
          <td>0.154785</td>
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
          <td>26.323830</td>
          <td>0.118910</td>
          <td>25.385742</td>
          <td>0.045739</td>
          <td>25.131991</td>
          <td>0.059685</td>
          <td>24.976311</td>
          <td>0.099362</td>
          <td>25.063761</td>
          <td>0.235021</td>
          <td>0.043542</td>
          <td>0.042334</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.662498</td>
          <td>0.159214</td>
          <td>26.030179</td>
          <td>0.080975</td>
          <td>25.253444</td>
          <td>0.066472</td>
          <td>24.889152</td>
          <td>0.092045</td>
          <td>24.201492</td>
          <td>0.112733</td>
          <td>0.047388</td>
          <td>0.026429</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.794704</td>
          <td>0.471922</td>
          <td>26.716210</td>
          <td>0.166678</td>
          <td>26.588912</td>
          <td>0.131993</td>
          <td>26.402828</td>
          <td>0.180815</td>
          <td>25.733482</td>
          <td>0.190795</td>
          <td>25.340494</td>
          <td>0.294634</td>
          <td>0.085606</td>
          <td>0.067795</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.995397</td>
          <td>0.251887</td>
          <td>26.247356</td>
          <td>0.111255</td>
          <td>26.097515</td>
          <td>0.085927</td>
          <td>25.893548</td>
          <td>0.116713</td>
          <td>25.791980</td>
          <td>0.200423</td>
          <td>25.046713</td>
          <td>0.231728</td>
          <td>0.036149</td>
          <td>0.031625</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.391204</td>
          <td>0.720944</td>
          <td>26.493418</td>
          <td>0.137708</td>
          <td>26.426719</td>
          <td>0.114658</td>
          <td>26.399254</td>
          <td>0.180269</td>
          <td>25.765810</td>
          <td>0.196062</td>
          <td>26.652403</td>
          <td>0.775746</td>
          <td>0.062642</td>
          <td>0.037997</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


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
          <td>27.978937</td>
          <td>1.129726</td>
          <td>26.812880</td>
          <td>0.207484</td>
          <td>26.090078</td>
          <td>0.100344</td>
          <td>25.207048</td>
          <td>0.075610</td>
          <td>24.565654</td>
          <td>0.081403</td>
          <td>24.040011</td>
          <td>0.115580</td>
          <td>0.001189</td>
          <td>0.000630</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.321378</td>
          <td>0.316298</td>
          <td>26.680070</td>
          <td>0.168215</td>
          <td>26.367357</td>
          <td>0.207570</td>
          <td>25.885251</td>
          <td>0.253792</td>
          <td>25.028895</td>
          <td>0.268617</td>
          <td>0.054283</td>
          <td>0.030536</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.739967</td>
          <td>0.886098</td>
          <td>27.685523</td>
          <td>0.386601</td>
          <td>26.023797</td>
          <td>0.156973</td>
          <td>24.854104</td>
          <td>0.106771</td>
          <td>24.364088</td>
          <td>0.155724</td>
          <td>0.081860</td>
          <td>0.064210</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.485589</td>
          <td>0.415199</td>
          <td>28.129566</td>
          <td>0.583614</td>
          <td>27.223831</td>
          <td>0.264990</td>
          <td>26.348962</td>
          <td>0.204481</td>
          <td>25.659600</td>
          <td>0.210609</td>
          <td>25.334397</td>
          <td>0.343371</td>
          <td>0.046510</td>
          <td>0.044264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.537462</td>
          <td>0.453166</td>
          <td>26.062783</td>
          <td>0.117235</td>
          <td>26.118941</td>
          <td>0.111330</td>
          <td>25.692970</td>
          <td>0.125559</td>
          <td>25.824320</td>
          <td>0.258296</td>
          <td>24.723566</td>
          <td>0.223980</td>
          <td>0.162684</td>
          <td>0.154785</td>
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
          <td>26.091451</td>
          <td>0.304771</td>
          <td>26.541674</td>
          <td>0.165916</td>
          <td>25.386791</td>
          <td>0.054265</td>
          <td>25.093443</td>
          <td>0.068831</td>
          <td>24.883281</td>
          <td>0.108245</td>
          <td>24.722713</td>
          <td>0.208517</td>
          <td>0.043542</td>
          <td>0.042334</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.934949</td>
          <td>1.104240</td>
          <td>26.637290</td>
          <td>0.179751</td>
          <td>26.206595</td>
          <td>0.111651</td>
          <td>25.202832</td>
          <td>0.075722</td>
          <td>24.776726</td>
          <td>0.098489</td>
          <td>24.140630</td>
          <td>0.126774</td>
          <td>0.047388</td>
          <td>0.026429</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.031374</td>
          <td>0.624558</td>
          <td>26.796636</td>
          <td>0.208214</td>
          <td>26.324635</td>
          <td>0.125558</td>
          <td>26.810580</td>
          <td>0.302535</td>
          <td>26.773962</td>
          <td>0.513703</td>
          <td>25.567899</td>
          <td>0.416528</td>
          <td>0.085606</td>
          <td>0.067795</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.150794</td>
          <td>0.319048</td>
          <td>26.263959</td>
          <td>0.130463</td>
          <td>26.013868</td>
          <td>0.094226</td>
          <td>25.943473</td>
          <td>0.144437</td>
          <td>25.659921</td>
          <td>0.210054</td>
          <td>25.681297</td>
          <td>0.447603</td>
          <td>0.036149</td>
          <td>0.031625</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.114197</td>
          <td>0.310973</td>
          <td>27.212283</td>
          <td>0.290377</td>
          <td>26.689736</td>
          <td>0.170014</td>
          <td>26.091470</td>
          <td>0.164808</td>
          <td>26.180037</td>
          <td>0.322838</td>
          <td>25.712076</td>
          <td>0.460160</td>
          <td>0.062642</td>
          <td>0.037997</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


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
          <td>26.577328</td>
          <td>0.148017</td>
          <td>25.996520</td>
          <td>0.078606</td>
          <td>25.120256</td>
          <td>0.059068</td>
          <td>24.696906</td>
          <td>0.077705</td>
          <td>23.973666</td>
          <td>0.092353</td>
          <td>0.001189</td>
          <td>0.000630</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.197151</td>
          <td>1.198819</td>
          <td>27.926322</td>
          <td>0.452352</td>
          <td>26.814594</td>
          <td>0.164282</td>
          <td>26.204380</td>
          <td>0.156710</td>
          <td>26.062117</td>
          <td>0.256891</td>
          <td>25.777466</td>
          <td>0.425070</td>
          <td>0.054283</td>
          <td>0.030536</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.886225</td>
          <td>1.731395</td>
          <td>28.111147</td>
          <td>0.535260</td>
          <td>28.566242</td>
          <td>0.671627</td>
          <td>26.021776</td>
          <td>0.139938</td>
          <td>24.964671</td>
          <td>0.105266</td>
          <td>24.411714</td>
          <td>0.145011</td>
          <td>0.081860</td>
          <td>0.064210</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.487557</td>
          <td>0.288123</td>
          <td>26.136384</td>
          <td>0.148099</td>
          <td>25.773132</td>
          <td>0.202469</td>
          <td>25.153512</td>
          <td>0.259821</td>
          <td>0.046510</td>
          <td>0.044264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.332451</td>
          <td>0.392391</td>
          <td>26.019370</td>
          <td>0.114728</td>
          <td>25.802864</td>
          <td>0.085905</td>
          <td>25.752887</td>
          <td>0.134638</td>
          <td>25.440328</td>
          <td>0.190878</td>
          <td>25.326996</td>
          <td>0.370594</td>
          <td>0.162684</td>
          <td>0.154785</td>
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
          <td>26.823140</td>
          <td>0.489000</td>
          <td>26.557468</td>
          <td>0.148568</td>
          <td>25.412683</td>
          <td>0.048020</td>
          <td>24.916022</td>
          <td>0.050568</td>
          <td>24.705589</td>
          <td>0.080249</td>
          <td>24.548554</td>
          <td>0.156016</td>
          <td>0.043542</td>
          <td>0.042334</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.418773</td>
          <td>0.742023</td>
          <td>26.778733</td>
          <td>0.178676</td>
          <td>25.925356</td>
          <td>0.075279</td>
          <td>25.247674</td>
          <td>0.067514</td>
          <td>24.836535</td>
          <td>0.089616</td>
          <td>24.252335</td>
          <td>0.120222</td>
          <td>0.047388</td>
          <td>0.026429</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.651966</td>
          <td>0.888380</td>
          <td>26.931544</td>
          <td>0.212817</td>
          <td>26.526073</td>
          <td>0.134539</td>
          <td>26.274111</td>
          <td>0.174803</td>
          <td>25.874690</td>
          <td>0.230487</td>
          <td>25.439057</td>
          <td>0.341911</td>
          <td>0.085606</td>
          <td>0.067795</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.065458</td>
          <td>0.269395</td>
          <td>26.268905</td>
          <td>0.114880</td>
          <td>26.041471</td>
          <td>0.083063</td>
          <td>25.997435</td>
          <td>0.129788</td>
          <td>25.567885</td>
          <td>0.168295</td>
          <td>25.194655</td>
          <td>0.265662</td>
          <td>0.036149</td>
          <td>0.031625</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.154918</td>
          <td>1.928141</td>
          <td>26.477514</td>
          <td>0.139952</td>
          <td>26.629460</td>
          <td>0.141465</td>
          <td>25.963053</td>
          <td>0.128540</td>
          <td>26.252299</td>
          <td>0.302371</td>
          <td>25.899460</td>
          <td>0.469977</td>
          <td>0.062642</td>
          <td>0.037997</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


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
